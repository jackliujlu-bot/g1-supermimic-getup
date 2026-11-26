"""Fine-tune the SuperMimic tracker: same CLI as supermimic/scripts/train.py, plus
--init DIR (world_model.pt + actor.pt [+ critic.pt]) to start from a trained tracker.

Example:
  CUDA_VISIBLE_DEVICES=1 python finetune_tracker.py SuperMimic-GenCoord-G1-Robust --init final/tracker \
      --init-critic <run>/critic.pt --num-envs 4096 --motion-file data/pack_ft \
      --runner.actor-obs proprio --runner.history-length 10 --runner.max-iterations 20000
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robust_task  # noqa: E402,F401
from supermimic.runners import supermimic_runner as R  # noqa: E402
from supermimic.scripts import train as T  # noqa: E402


def _pop(flag):
  if flag in sys.argv:
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v
  return None


init_dir, init_critic = _pop("--init"), _pop("--init-critic")
teacher_dir, bc_weight = _pop("--teacher"), float(_pop("--bc-weight") or 0.0)
_orig_init = R.SuperMimicRunner.__init__


def _init(self, *a, **kw):
  _orig_init(self, *a, **kw)
  dev = self.device if hasattr(self, "device") else "cuda:0"
  if init_dir:
    self._wm.load_bundle(torch.load(os.path.join(init_dir, "world_model.pt"), map_location=dev))
    self.policy_learner.actor.load_state_dict(torch.load(os.path.join(init_dir, "actor.pt"), map_location=dev))
    print(f"[finetune] loaded world model + actor from {init_dir}")
  crit = getattr(self.policy_learner, "critic", None)
  if init_critic and crit is not None:
    crit.load_state_dict(torch.load(init_critic, map_location=dev))
    print(f"[finetune] loaded critic from {init_critic}")


  if teacher_dir and bc_weight > 0:
    _add_teacher_bc(self.policy_learner.actor, teacher_dir, bc_weight, dev)


def _add_teacher_bc(actor, teacher_dir, w, dev):
    """Keep the fine-tuned tracker close to the original one: add w * |a - a_teacher|^2 to the
    actor's auxiliary loss on every imagined step (the teacher is frozen; it is the original,
    dance-proven tracker). The return term still dominates where the teacher fails (get-up on a
    soft mat), so the tracker can change there while dances stay as they were."""
    import copy

    teacher = copy.deepcopy(actor)
    teacher.load_state_dict(torch.load(os.path.join(teacher_dir, "actor.pt"), map_location=dev))
    teacher.eval().requires_grad_(False)
    orig_forward = actor.forward
    object.__setattr__(actor, "_bc_teacher", [teacher])  # list: not registered as a submodule

    def forward(obs_td):
        a = orig_forward(obs_td)
        if actor.training:
            with torch.no_grad():
                a_t = actor._bc_teacher[0](obs_td)
            actor._aux_loss = actor._aux_loss + w * ((a - a_t) ** 2).mean(dim=-1)
        return a

    actor.forward = forward
    print(f"[finetune] teacher BC to {teacher_dir}, weight {w}")


R.SuperMimicRunner.__init__ = _init
if __name__ == "__main__":
  T.main()
