import numpy as np
import torch

from shared_backbone import BackboneConfig, SharedBackbone


torch.manual_seed(0)
rng = np.random.default_rng(0)


def calm_frame() -> np.ndarray:

    base = rng.integers(95, 105)
    return np.clip(base + rng.integers(-8, 8, size=(480, 640, 3)), 0, 255).astype(np.uint8)


def weird_frame() -> np.ndarray:

    img = calm_frame().astype(np.int16)
    img[:, :, 0] += 135
    img[:, :, 2] -= 70
    return np.clip(img, 0, 255).astype(np.uint8)


def line(c="-"):
    print(c * 64)


def main() -> None:

    cfg = BackboneConfig(pretrained=False, input_size=256, freeze_encoder=True, gate_warmup=6)
    bb = SharedBackbone(cfg)


    line("=")
    print("1) PARÁMETROS  (encoder congelado: etapa 1)")
    line()
    rep = bb.param_report()
    print(f"   total      : {rep['total']/1e6:6.1f} M")
    print(f"   entrenables: {rep['trainable']/1e6:6.1f} M   (FPN + embed + GRU)")
    print(f"   congelados : {rep['frozen']/1e6:6.1f} M   (ResNet-50)")


    line("=")
    print("2) QUÉ PRODUCE  (un frame de cam-3)")
    line()
    feats = bb.encode(calm_frame(), camera_id="cam-3")
    for lvl, t in feats.pyramid.items():
        print(f"   pyramid[{lvl}]: {tuple(t.shape)}  stride {feats.strides[lvl]}")
    print(f"   embedding  : {tuple(feats.embedding.shape)}")
    print(f"   temporal   : {feats.temporal}  (None en el 1er frame, buffer frío)")
    print(f"   novelty    : {feats.novelty:.3f}")
    print(f"   meta       : cam={feats.camera_id} idx={feats.frame_idx} "
          f"scale p3->orig x{feats.scale_xy('p3'):.1f}")


    line("=")
    print("3) TEMPORAL  (se llena al acumular frames de la misma cámara)")
    line()
    for i in range(3):
        f = bb.encode(calm_frame(), camera_id="cam-3")
        shape = tuple(f.temporal.shape) if f.temporal is not None else None
        print(f"   frame idx={f.frame_idx}: temporal={shape}")


    line("=")
    print("4) GATE DEL VLM  (novedad de la escena por cámara)")
    line()
    bb.reset("cam-3")
    print("   alimentamos escena tranquila (aprende lo 'normal')...")
    for _ in range(8):
        f = bb.encode(calm_frame(), camera_id="cam-3")
    print(f"   tranquila -> novelty={f.novelty:.1f}σ  vlm={bb.should_run_vlm(f)}")

    f = bb.encode(weird_frame(), camera_id="cam-3")
    print(f"   RARA      -> novelty={f.novelty:.1f}σ  vlm={bb.should_run_vlm(f)}")

    f2 = bb.encode(calm_frame(), camera_id="cam-3")
    print(f"   por duda  -> (cabezas inseguras 0.6) vlm="
          f"{bb.should_run_vlm(f2, head_uncertainty=0.6)}")


    line("=")
    print("5) MULTI-CÁMARA  (estado independiente por cámara)")
    line()
    bb.reset()
    for cam in ("cam-1", "cam-2"):
        for _ in range(3):
            f = bb.encode(calm_frame(), camera_id=cam)
        print(f"   {cam}: idx={f.frame_idx} (cuenta sus propios frames)")


    line("=")
    print("6) FREEZE / UNFREEZE  (etapa 1 -> etapa 2)")
    line()
    bb.set_encoder_frozen(False)
    print(f"   encoder descongelado -> entrenables: {bb.param_report()['trainable']/1e6:.1f} M")
    bb.set_encoder_frozen(True)
    print(f"   encoder congelado    -> entrenables: {bb.param_report()['trainable']/1e6:.1f} M")

    line("=")
    print("OK ✅  backbone compartido funcionando con todas sus funciones.")


if __name__ == "__main__":
    main()
