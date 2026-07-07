from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import torch

from shared_backbone import BackboneConfig, SharedBackbone


def percentil(valores, p):

    s = sorted(valores)
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * len(s))) - 1))
    return s[i]


def sync(device):

    if device == "cuda":
        torch.cuda.synchronize()


def bench_encode(frame, input_size, warmup, frames, pretrained, device):

    bb = SharedBackbone(
        BackboneConfig(pretrained=pretrained, input_size=input_size, freeze_encoder=True)
    ).to(device).eval()

    for _ in range(warmup):
        bb.encode(frame, camera_id="bench")
    sync(device)

    lat_ms = []
    for _ in range(frames):
        t0 = time.perf_counter()
        bb.encode(frame, camera_id="bench")
        sync(device)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    media = sum(lat_ms) / len(lat_ms)
    return {"media_ms": media, "p95_ms": percentil(lat_ms, 95), "fps": 1000.0 / media}


def bench_batch(input_size, batch, warmup, frames, pretrained, device):

    bb = SharedBackbone(
        BackboneConfig(pretrained=pretrained, input_size=input_size, freeze_encoder=True)
    ).to(device).eval()
    x = torch.randn(batch, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            bb(x, camera_id="b", update_temporal=False)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(frames):
            bb(x, camera_id="b", update_temporal=False)
        sync(device)
        dt = time.perf_counter() - t0
    return (batch * frames) / dt


def main():
    ap = argparse.ArgumentParser(description="F0: medir FPS del backbone.")
    ap.add_argument("--cameras", type=int, default=4, help="Cuántas cámaras pensás correr.")
    ap.add_argument("--target-fps", type=float, default=5.0, help="FPS por cámara objetivo.")
    ap.add_argument("--input-sizes", type=int, nargs="+", default=[256, 320, 384])
    ap.add_argument("--frames", type=int, default=60, help="Frames medidos por tamaño.")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--frame-width", type=int, default=640)
    ap.add_argument("--frame-height", type=int, default=480)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    frame = rng.integers(0, 256, (args.frame_height, args.frame_width, 3), dtype=np.uint8)

    print("=" * 66)
    print("F0 · MEDICIÓN DE FPS DEL BACKBONE")
    print("=" * 66)
    print(f"  dispositivo : {device}", end="")
    if device == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  ({platform.processor() or 'CPU'}, {torch.get_num_threads()} hilos)")
    print(f"  torch       : {torch.__version__}")
    print(f"  frame real  : {args.frame_width}x{args.frame_height}")
    print(f"  objetivo    : {args.cameras} cámaras a {args.target_fps:g} FPS c/u")
    print(f"  carga total : {args.cameras * args.target_fps:g} frames/seg\n")


    print("Costo por frame de 1 cámara (encode: preproceso + backbone):")
    print(f"  {'tamaño':>7} | {'latencia':>9} | {'p95':>7} | {'FPS/cám':>8} | {'cámaras a ' + format(args.target_fps, 'g') + ' FPS':>16}")
    print("  " + "-" * 60)
    mejor = None
    for size in args.input_sizes:
        r = bench_encode(frame, size, args.warmup, args.frames, args.pretrained, device)
        cams = r["fps"] / args.target_fps
        marca = "  <- alcanza" if cams >= args.cameras else ""
        print(f"  {size:>5}px | {r['media_ms']:>7.1f}ms | {r['p95_ms']:>5.1f}ms "
              f"| {r['fps']:>6.1f}   | {cams:>6.1f}{marca}")
        if mejor is None or size == 384 or (384 not in args.input_sizes and size == args.input_sizes[-1]):
            mejor = (size, r, cams)


    size_ref = mejor[0]
    print(f"\nThroughput por batch a {size_ref}px (procesar varias cámaras juntas):")
    for b in (1, 2, 4):
        ips = bench_batch(size_ref, b, args.warmup, max(20, args.frames // 2),
                          args.pretrained, device)
        print(f"  batch {b}: {ips:6.1f} imágenes/seg  ({ips/args.target_fps:4.1f} cámaras a "
              f"{args.target_fps:g} FPS)")


    size_v, r_v, cams_v = mejor
    print("\n" + "=" * 66)
    print("VEREDICTO")
    print("=" * 66)
    if cams_v >= args.cameras:
        print(f"  ✅ A {size_v}px tu hardware aguanta ~{cams_v:.1f} cámaras a "
              f"{args.target_fps:g} FPS;\n     necesitás {args.cameras}. Alcanza, podés usar este backbone.")
    else:
        print(f"  ⚠️  A {size_v}px tu hardware aguanta ~{cams_v:.1f} cámaras a "
              f"{args.target_fps:g} FPS,\n     pero necesitás {args.cameras}. Opciones:")
        print("     - bajar el input_size (probá los tamaños más chicos de la tabla),")
        print("     - bajar el FPS objetivo o las cámaras por equipo,")
        print("     - procesar las cámaras en batch (ver throughput de arriba),")
        print("     - usar GPU, o un backbone más liviano (ResNet-18/34, MobileNet).")
    print("\n  Nota: la velocidad no depende de los pesos; corré esto en el equipo del cliente.")


if __name__ == "__main__":
    main()
