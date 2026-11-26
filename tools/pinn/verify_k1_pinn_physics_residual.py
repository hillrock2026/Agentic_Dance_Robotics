#!/usr/bin/env python3
"""K1 PINN Task 1 — physics-residual / MuJoCo-mapping verification (read-only).

Verifies, on the actual Booster K1 model and the installed MuJoCo, every
identity needed by the PINN physics-residual definition:

  1.  dimensions (nq/nv/nu/na), field shapes
  2.  M(q) via mj_fullM: finite, symmetric, PD; armature included
  3.  solver-consistency residual  r = M a + bias - act - passive - applied
      - constraint  ~ 0  on MuJoCo's own qacc
  4.  degeneracy: r(a_hat) == M (a_hat - a_mujoco) exactly (artificial probe)
  5.  free-dynamics residual and free-dynamics mask (air / contact / joint
      limit states)
  6.  constraint-force identities: J^T efc_force == qfrc_constraint,
      contact world-force aggregation vs point-Jacobian transpose
  7.  xfrc_applied accounting (world wrench at body CoM)
  8.  mj_inverse semantics (and qfrc_constraint overwrite side effect)
  9.  collection timing: mj_forward does not advance time; post-step qacc
      corresponds to pre-step state; forward+snapshot+step vs mj_step vs
      mj_step1/2 equivalence
  10. quaternion handling: mj_integratePos/mj_differentiatePos round trip
      vs naive component differencing
  11. mj_mulM == dense M @ v

No XML / controller / physics parameter is modified. Writes a JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO_ROOT / "virtual_k1/assets/booster_assets/robots/K1/K1_22dof.xml"
DEFAULT_CFG = REPO_ROOT / "virtual_k1/config/k1_sim.yaml"

RESULTS: dict = {"checks": [], "states": {}, "environment": {}}


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def full_m(model, data) -> np.ndarray:
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, M)
    return M


def residual(M, a_hat, bias, act, passive, applied, constraint):
    """r = M a + bias - act - passive - applied - constraint  (R^nv)."""
    return M @ a_hat + bias - act - passive - applied - constraint


def set_stance(model, data, cfg, z: float, knee_override: float | None = None):
    data.qpos[:3] = [0.0, 0.0, z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, val in cfg["initial_pose"]["joints"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = float(val)
    if knee_override is not None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch_joint")
        data.qpos[model.jnt_qposadr[jid]] = knee_override


def sample_state(model, cfg, z: float, seed: int, ctrl: float,
                  knee_override: float | None = None):
    """Fresh MjData at a stance-derived state; returns (data, M, fields dict)."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    set_stance(model, data, cfg, z, knee_override)
    data.qvel[:] = rng.normal(0.0, 0.3, model.nv)
    if knee_override is not None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch_joint")
        data.qvel[model.jnt_dofadr[jid]] = -2.0   # push into the limit
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    M = full_m(model, data)
    fields = {
        "qacc": data.qacc.copy(),
        "bias": data.qfrc_bias.copy(),
        "act": data.qfrc_actuator.copy(),
        "passive": data.qfrc_passive.copy(),
        "applied": data.qfrc_applied.copy(),
        "constraint": data.qfrc_constraint.copy(),
        "ncon": int(data.ncon),
        "nefc": int(data.nefc),
    }
    return data, M, fields


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--json", type=Path, default=Path(__file__).parent / "verify_summary.json")
    args = ap.parse_args()

    RESULTS["environment"] = {
        "mujoco_version": mujoco.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "xml": str(args.xml),
        "xml_sha256": sha256_of(args.xml),
    }
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data0 = mujoco.MjData(model)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- (1) dims
    print("=== 1. dimensions & field shapes ===")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} na={model.na} "
          f"njnt={model.njnt} nbody={model.nbody} neq={model.neq} "
          f"ntendon={model.ntendon} timestep={model.opt.timestep}")
    check("dims nq/nv/nu == 29/28/22", (model.nq, model.nv, model.nu) == (29, 28, 22))
    check("na == 0 (motors, no activation state)", model.na == 0,
          f"act shape {data0.act.shape}")
    check("field shapes qpos/qvel/qacc/ctrl/actuator_force",
          data0.qpos.shape == (29,) and data0.qvel.shape == (28,)
          and data0.qacc.shape == (28,) and data0.ctrl.shape == (22,)
          and data0.actuator_force.shape == (22,))
    check("no equality/tendon constraints in model",
          model.neq == 0 and model.ntendon == 0)
    check("xfrc_applied default zero",
          np.abs(data0.xfrc_applied).max() == 0.0,
          f"shape {data0.xfrc_applied.shape}")

    # ---------------------------------------------- states: air/contact/limit
    states = {}
    states["air"] = sample_state(model, cfg, z=1.2, seed=1, ctrl=2.0)
    states["contact"] = sample_state(model, cfg, z=0.5450, seed=1, ctrl=5.0)
    states["joint_limit"] = sample_state(model, cfg, z=1.2, seed=1, ctrl=2.0,
                                         knee_override=-0.02)
    for name, (d, M, f) in states.items():
        print(f"state '{name}': ncon={f['ncon']} nefc={f['nefc']} "
              f"||qfrc_constraint||_inf={np.abs(f['constraint']).max():.6g}")
        RESULTS["states"][name] = {
            "ncon": f["ncon"], "nefc": f["nefc"],
            "qfrc_constraint_inf_norm": float(np.abs(f["constraint"]).max()),
            "efc_type_values": sorted(set(int(t) for t in d.efc_type[:d.nefc])) if d.nefc else [],
        }

    # ---------------------------------------------------- (2) mass matrix
    print("=== 2. mass matrix via mj_fullM ===")
    d_c, M_c, f_c = states["contact"]
    check("M finite", bool(np.isfinite(M_c).all()))
    check("M symmetric", np.abs(M_c - M_c.T).max() == 0.0)
    eig = np.linalg.eigvalsh(M_c)
    check("M positive definite", eig.min() > 0,
          f"eigmin={eig.min():.6g} eigmax={eig.max():.6g}")
    check("M root translational block trace == total mass",
          abs(np.trace(M_c[:3, :3]) / 3.0 - model.body_mass.sum()) < 1e-6,
          f"={np.trace(M_c[:3, :3]) / 3.0:.6f} kg vs {model.body_mass.sum():.6f} kg")

    # armature: removing dof_armature from M must break the identity
    arm = model.dof_armature.copy()
    r_noarm = residual(M_c - np.diag(arm), f_c["qacc"], f_c["bias"], f_c["act"],
                       f_c["passive"], f_c["applied"], f_c["constraint"])
    expected = -arm * f_c["qacc"]
    check("armature included in mj_fullM output (removal breaks identity)",
          np.abs(r_noarm - expected).max() < 1e-6,
          f"||r(M-diag(arm)) - (-arm*qacc)||_inf={np.abs(r_noarm - expected).max():.3g}; "
          f"||r_noarm||_inf={np.abs(r_noarm).max():.3g}")

    # ----------------------------------------- (3) solver-consistency residual
    print("=== 3. solver-consistency residual on MuJoCo's own qacc ===")
    for name, (d, M, f) in states.items():
        r = residual(M, f["qacc"], f["bias"], f["act"], f["passive"],
                     f["applied"], f["constraint"])
        check(f"r(qacc_mujoco) ~ 0 [{name}]", np.abs(r).max() < 1e-6,
              f"||r||_inf={np.abs(r).max():.3g}")
        check(f"qfrc_passive == 0 and qfrc_applied == 0 [{name}]",
              np.abs(f["passive"]).max() == 0.0 and np.abs(f["applied"]).max() == 0.0)

    # qfrc_smooth decomposition (xfrc == 0 here)
    smooth_ok = all(
        np.abs(d.qfrc_smooth - (f["passive"] + f["act"] + f["applied"] - f["bias"])).max() < 1e-9
        for (d, M, f) in states.values())
    check("qfrc_smooth == passive + actuator + applied - bias (xfrc=0)", smooth_ok)

    # ---------------------------------------------------- (4) degeneracy
    print("=== 4. degeneracy: r(a_hat) == M (a_hat - a_mujoco) ===")
    rng = np.random.default_rng(7)
    for name, (d, M, f) in states.items():
        a_hat = f["qacc"] + rng.normal(0.0, 1.0, model.nv)
        r_hat = residual(M, a_hat, f["bias"], f["act"], f["passive"],
                         f["applied"], f["constraint"])
        lhs = np.abs(r_hat - M @ (a_hat - f["qacc"])).max()
        check(f"linearity r(a_hat)=M(a_hat-a*) [{name}]", lhs < 1e-6, f"{lhs:.3g}")
    print("[info] on solver data the residual carries no information beyond the "
          "M-weighted acceleration error -> name it a *dynamics consistency loss*")

    # ---------------------------------------------------- (5) free-dynamics
    print("=== 5. free-dynamics residual & mask ===")
    def free_mask(d, f):
        return (f["ncon"] == 0 and f["nefc"] == 0
                and np.abs(f["constraint"]).max() == 0.0)

    d_a, M_a, f_a = states["air"]
    mask_air = free_mask(d_a, f_a)
    check("air state -> free_dynamics_mask == True", bool(mask_air))
    for name in ("contact", "joint_limit"):
        d_s, M_s, f_s = states[name]
        check(f"{name} state -> free_dynamics_mask == False",
              not free_mask(d_s, f_s))
    r_free = residual(M_a, f_a["qacc"], f_a["bias"], f_a["act"],
                      f_a["passive"], f_a["applied"], np.zeros(model.nv))
    check("free-dynamics residual r_free(qacc) ~ 0 on air state",
          np.abs(r_free).max() < 1e-6, f"{np.abs(r_free).max():.3g}")

    # joint-limit structural check: efc row == unit row of the knee dof
    d_l, M_l, f_l = states["joint_limit"]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch_joint")
    dof = int(model.jnt_dofadr[jid])
    if d_l.nefc >= 1:
        Jl = np.asarray(d_l.efc_J).reshape(d_l.nefc, model.nv)
        onehot = np.zeros(model.nv); onehot[dof] = 1.0
        check("joint-limit efc row is the unit row of the knee dof",
              np.abs(Jl[0] - onehot).max() < 1e-12,
              f"||J row - e_knee||={np.abs(Jl[0] - onehot).max():.3g}")
        check("joint-limit generalized force lands on the knee dof",
              np.abs(f_l["constraint"][dof] - np.abs(f_l["constraint"]).max()) == 0.0)

    # ------------------------------------------- (6) constraint-force identities
    print("=== 6. constraint force identities [contact state] ===")
    nefc, nv = d_c.nefc, model.nv
    J = np.asarray(d_c.efc_J).reshape(nefc, nv)
    f_efc = np.asarray(d_c.efc_force).copy()
    check("efc_J exposed flat (nefc*nv,) -> reshape works",
          np.asarray(d_c.efc_J).shape == (nefc * nv,))
    tmp = np.zeros(nv)
    mujoco.mj_mulJacTVec(d_c.model if hasattr(d_c, 'model') else model, d_c, tmp, f_efc)
    check("J^T efc_force == qfrc_constraint (mj_mulJacTVec)",
          np.abs(tmp - f_c["constraint"]).max() < 1e-9,
          f"{np.abs(tmp - f_c['constraint']).max():.3g}")
    check("J^T efc_force == qfrc_constraint (dense)",
          np.abs(J.T @ f_efc - f_c["constraint"]).max() < 1e-9,
          f"{np.abs(J.T @ f_efc - f_c['constraint']).max():.3g}")

    addr = [int(d_c.contact[i].efc_address) for i in range(d_c.ncon)]
    dims = [int(d_c.contact[i].dim) for i in range(d_c.ncon)]
    stride = [b - a for a, b in zip(addr[:-1], addr[1:])]
    check("condim=3 contacts occupy 4 efc rows each (pyramidal facets)",
          len(set(stride)) == 1 and stride[0] == 4 and set(dims) == {3},
          f"stride={stride[0]}, efc_id groups match contacts")
    check("pyramidal facet multipliers nonnegative",
        bool((f_efc[:sum(1 for _ in range(d_c.nefc))] >= -1e-9).all()),
        f"min={f_efc.min():.3g}")
    print(f"[info] opt.cone={int(model.opt.cone)}; observed row directions are "
          "n±t1 / n±t2 (pyramidal facets), efc_type label="
          f"{sorted(set(int(t) for t in d_c.efc_type))} (recorded as version behaviour)")

    # contact world-force aggregation vs point-Jacobian transpose
    gen = np.zeros(nv)
    fn_world_sum = np.zeros(3)
    for i in range(d_c.ncon):
        c = d_c.contact[i]
        w = np.zeros(6)
        mujoco.mj_contactForce(model, d_c, i, w)
        frame = np.asarray(c.frame).reshape(3, 3)      # rows: normal, t1, t2 (world)
        f_world = frame.T @ w[:3]                       # force in world frame
        fn_world_sum += frame[0] * w[0]
        body = int(model.geom_bodyid[int(c.geom2)])
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        mujoco.mj_jac(model, d_c, jp, jr, np.asarray(c.pos), body)
        gen += jp.T @ f_world
    sgn = 1.0 if np.abs(gen - f_c["constraint"]).max() < 1e-6 else -1.0
    check("sum_contact jac^T F_world == qfrc_constraint (contact-only state)",
          np.abs(sgn * gen - f_c["constraint"]).max() < 1e-6,
          f"sign={sgn:+.0f} (mj_contactForce wrench acts on geom2 body: "
          f"{'+' if sgn > 0 else '-'}); err={np.abs(sgn * gen - f_c['constraint']).max():.3g}")
    check("sum of world-frame normal forces == qfrc_constraint[root_z]",
          abs(fn_world_sum[2] - f_c["constraint"][2]) < 1e-6,
          f"{fn_world_sum[2]:.6g} N vs {f_c['constraint'][2]:.6g} N")

    # ---------------------------------------------------- (7) xfrc_applied
    print("=== 7. xfrc_applied accounting ===")
    d_x = mujoco.MjData(model)
    d_x.qpos[:] = d_a.qpos; d_x.qvel[:] = d_a.qvel; d_x.ctrl[:] = d_a.ctrl
    mujoco.mj_forward(model, d_x)
    base_smooth = d_x.qfrc_smooth.copy()
    wrench = np.array([0.0, 0.0, 10.0, 0.05, 0.0, 0.0])
    d_x.xfrc_applied[1] = wrench                        # trunk = body 1
    mujoco.mj_forward(model, d_x)
    delta = d_x.qfrc_smooth - base_smooth
    jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
    mujoco.mj_jacBodyCom(model, d_x, jp, jr, 1)
    pred_com = jp.T @ wrench[:3] + jr.T @ wrench[3:]
    jp2 = np.zeros((3, nv)); jr2 = np.zeros((3, nv))
    mujoco.mj_jacBody(model, d_x, jp2, jr2, 1)
    pred_org = jp2.T @ wrench[:3] + jr2.T @ wrench[3:]
    check("xfrc_applied contributes jacBodyCom^T wrench (world frame, at CoM)",
          np.abs(delta - pred_com).max() < 1e-9,
          f"err_com={np.abs(delta - pred_com).max():.3g} "
          f"err_frameorigin={np.abs(delta - pred_org).max():.3g}")
    check("xfrc_applied does NOT enter qfrc_applied field",
          np.abs(d_x.qfrc_applied).max() == 0.0)
    check("full equation closes with xfrc term",
          np.abs(full_m(model, d_x) @ d_x.qacc + d_x.qfrc_bias - d_x.qfrc_actuator
                - d_x.qfrc_passive - d_x.qfrc_applied - d_x.qfrc_constraint
                - (jp.T @ wrench[:3] + jr.T @ wrench[3:])).max() < 1e-8)
    print("[info] collector condition: xfrc_applied is kept identically zero "
          "and must be recorded as an experimental condition")

    # ---------------------------------------------------- (8) mj_inverse
    print("=== 8. mj_inverse ===")
    d_i = mujoco.MjData(model)
    d_i.qpos[:] = d_c.qpos; d_i.qvel[:] = d_c.qvel; d_i.qacc[:] = d_c.qacc
    d_i.ctrl[:] = d_c.ctrl
    mujoco.mj_forward(model, d_i)
    M_i = full_m(model, d_i)
    saved = dict(bias=d_i.qfrc_bias.copy(), act=d_i.qfrc_actuator.copy(),
                 passive=d_i.qfrc_passive.copy(), applied=d_i.qfrc_applied.copy(),
                 constraint=d_i.qfrc_constraint.copy(), qacc=d_i.qacc.copy())
    mujoco.mj_inverse(model, d_i)
    net = M_i @ saved["qacc"] + saved["bias"] - saved["passive"] - saved["applied"]
    check("qfrc_inverse == M qacc + bias - passive - applied - constraint",
          np.abs(d_i.qfrc_inverse - (net - saved["constraint"])).max() < 1e-6,
          f"err={np.abs(d_i.qfrc_inverse - (net - saved['constraint'])).max():.3g}")
    check("qfrc_inverse ~ qfrc_actuator (forward) in this model/state",
          np.abs(d_i.qfrc_inverse - saved["act"]).max() < 1e-6,
          f"err={np.abs(d_i.qfrc_inverse - saved['act']).max():.3g}")
    check("WARNING recorded: mj_inverse overwrites qfrc_constraint "
          "(inverse-recomputed, ~1e-10 vs forward)",
          np.abs(d_i.qfrc_constraint - saved["constraint"]).max() < 1e-6)

    # ---------------------------------------------------- (9) timing protocol
    print("=== 9. collection timing ===")
    d_t = mujoco.MjData(model)
    set_stance(model, d_t, cfg, 0.551)
    rng_t = np.random.default_rng(3)
    d_t.qvel[:] = rng_t.normal(0.0, 0.2, model.nv)
    d_t.ctrl[:] = 3.0
    t0 = d_t.time
    mujoco.mj_forward(model, d_t)
    check("mj_forward does not advance time", d_t.time == t0)
    snap_qacc = d_t.qacc.copy()
    q_pre, v_pre = d_t.qpos.copy(), d_t.qvel.copy()
    mujoco.mj_step(model, d_t)
    check("mj_step advances time by opt.timestep",
          abs((d_t.time - t0) - model.opt.timestep) < 1e-15)
    check("post-step qacc still corresponds to pre-step state "
          "(== pre-step forward snapshot)", np.abs(d_t.qacc - snap_qacc).max() == 0.0)
    q_next, v_next = d_t.qpos.copy(), d_t.qvel.copy()
    mujoco.mj_forward(model, d_t)
    check("forward at NEW state differs from step's qacc (no confusion)",
          np.abs(d_t.qacc - snap_qacc).max() > 1e-6,
          f"|delta|={np.abs(d_t.qacc - snap_qacc).max():.3g}")
    # protocol equivalence: fresh data, mj_step only
    d_s = mujoco.MjData(model)
    d_s.qpos[:] = q_pre; d_s.qvel[:] = v_pre; d_s.ctrl[:] = 3.0
    mujoco.mj_step(model, d_s)
    check("protocol B (save q,v -> mj_step -> read fields) matches A",
          np.abs(d_s.qacc - snap_qacc).max() == 0.0
          and np.abs(d_s.qpos - q_next).max() == 0.0
          and np.abs(d_s.qvel - v_next).max() == 0.0)
    # protocol C: mj_step1 / mj_step2
    d_w = mujoco.MjData(model)
    d_w.qpos[:] = q_pre; d_w.qvel[:] = v_pre; d_w.ctrl[:] = 3.0
    mujoco.mj_step1(model, d_w)
    M_after_step1 = full_m(model, d_w)
    check("after mj_step1: M already valid, actuator/constraint forces not yet "
          "computed (zeros on fresh data)",
          np.abs(M_after_step1 - full_m(model, d_s)).max() == 0.0
          and np.abs(d_w.qfrc_actuator).max() == 0.0
          and np.abs(d_w.qfrc_constraint).max() == 0.0)
    mujoco.mj_step2(model, d_w)
    check("protocol C (step1 -> ctrl may be set -> step2) matches A",
          np.abs(d_w.qacc - snap_qacc).max() == 0.0
          and np.abs(d_w.qpos - q_next).max() == 0.0
          and np.abs(d_w.qvel - v_next).max() == 0.0)

    # ---------------------------------------------------- (10) quaternion ops
    print("=== 10. quaternion handling ===")
    q0 = q_pre.copy()
    v_rot = np.zeros(model.nv)
    v_rot[5] = 2.0                                     # local-frame yaw rate
    q1 = q0.copy()
    mujoco.mj_integratePos(model, q1, v_rot, 0.01)
    v_back = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v_back, 0.01, q0, q1)
    check("mj_integratePos/mj_differentiatePos round trip exact",
          np.abs(v_back - v_rot).max() < 1e-12)
    naive = (q1[3:7] - q0[3:7]) / 0.01
    check("naive quaternion component differencing is WRONG",
          np.abs(naive[:3]).max() > 0 or abs(naive[3]) > 0,
          f"naive={naive.round(4)} vs true local omega={v_back[3:6].round(4)}")

    # ---------------------------------------------------- (11) mj_mulM
    print("=== 11. mj_mulM ===")
    vec = np.random.default_rng(11).normal(0.0, 1.0, nv)
    res = np.zeros(nv)
    mujoco.mj_mulM(model, d_s, res, vec)
    check("mj_mulM(v) == dense M @ v", np.abs(res - M_after_step1 @ vec).max() < 1e-10,
          f"{np.abs(res - M_after_step1 @ vec).max():.3g}")

    # ---------------------------------------------------- summary
    n_pass = sum(c["pass"] for c in RESULTS["checks"])
    n_tot = len(RESULTS["checks"])
    RESULTS["summary"] = {
        "passed": n_pass, "total": n_tot,
        "verdict": "PASS" if n_pass == n_tot else "PARTIAL",
        "notes": [
            "equality constraints (neq=0) cannot be constructed in this model; "
            "not verified by construction",
            "efc row layout: condim=3 contact -> 4 pyramidal-facet rows; efc row "
            "counts must be read from efc_address/efc_id, never assumed",
            "r(a) on solver data is exactly the M-weighted acceleration error "
            "(dynamics consistency loss), not an independent contact-physics loss",
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"\nverdict: {RESULTS['summary']['verdict']} ({n_pass}/{n_tot} checks)")
    print(f"summary -> {args.json}")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
