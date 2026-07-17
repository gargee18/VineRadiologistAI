import numpy as np
from vineradiology.projection import generate_drr, apply_yaw, apply_pitch, apply_roll, apply_distance, render_pose, Pose
from vineradiology.deform import bend_volume, elastic_deform


def test_generate_drr_range():
    vol = np.ones((10, 10, 10), dtype=np.float32)
    drr = generate_drr(vol, attenuation_scale=0.015, axis=1)
    assert drr.shape == (10, 10)
    assert np.all(drr >= 0) and np.all(drr <= 1)


def test_generate_drr_zero_volume_is_zero():
    vol = np.zeros((5, 5, 5), dtype=np.float32)
    drr = generate_drr(vol)
    assert np.allclose(drr, 0)


def test_apply_yaw_zero_is_identity():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    out = apply_yaw(vol, 0)
    assert np.allclose(out, vol, atol=1e-5)


def test_apply_pitch_shape_preserved():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    out = apply_pitch(vol, 15)
    assert out.shape == vol.shape


def test_apply_roll_shape_preserved():
    img = np.random.rand(20, 20).astype(np.float32)
    out = apply_roll(img, 10)
    assert out.shape == img.shape


def test_apply_distance_identity_at_one():
    img = np.random.rand(20, 20).astype(np.float32)
    out = apply_distance(img, 1.0)
    assert np.array_equal(out, img)


def test_apply_distance_shape_preserved():
    img = np.random.rand(20, 20).astype(np.float32)
    out = apply_distance(img, 1.2)
    assert out.shape == img.shape


def test_render_pose_shape():
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    pose = Pose(yaw=30, pitch=10, roll=5, distance=1.1, attenuation_scale=0.015)
    img = render_pose(vol, pose)
    assert img.shape == (16, 16)


def test_bend_volume_shape_preserved():
    vol = np.random.rand(12, 10, 10).astype(np.float32)
    out = bend_volume(vol, max_angle_deg=15, axis=0)
    assert out.shape == vol.shape


def test_elastic_deform_shape_preserved():
    vol = np.random.rand(10, 10, 10).astype(np.float32)
    out = elastic_deform(vol, alpha=50, sigma=4, random_state=np.random.default_rng(0))
    assert out.shape == vol.shape
