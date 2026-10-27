import torch

from temporal_mf.eval.rollout import rollout, teacher_forced


def test_rollout_uses_predictions_not_truth():
    def step(state):
        return state + 1

    u0 = torch.zeros(1, 4)
    out = rollout(step, u0, 3)
    assert torch.equal(out[0, 0], torch.zeros(4))
    assert torch.equal(out[0, 1], torch.ones(4))
    assert torch.equal(out[0, 3], torch.full((4,), 3.0))

    def sensitive(state):
        return state * 0 + state.mean()

    base = rollout(sensitive, torch.tensor([[0.0, 1.0, 2.0]]), 2)
    bumped = rollout(sensitive, torch.tensor([[0.0, 1.0, 5.0]]), 2)
    assert not torch.allclose(base[:, 1], bumped[:, 1])


def test_teacher_forced_reads_truth_frames():
    truth = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    def step(state):
        return state + 10

    preds = teacher_forced(step, truth)
    assert torch.allclose(preds[0, 0], truth[0, 0] + 10)
    assert torch.allclose(preds[0, 2], truth[0, 2] + 10)
