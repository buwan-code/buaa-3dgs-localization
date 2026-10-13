
#include <cstdio>
#include <vector>
#include "splat_loader.h"
int main(int argc, char** argv) {
    if (argc < 2) { printf("用法: %s model.splat\n", argv[0]); return 1; }
    if (!gsplat::looks_like_splat(argv[1])) { printf("不是 .splat\n"); return 1; }
    std::vector<gsplat::Splat> sp; std::vector<gsplat::ShCoeffs> sh;
    if (!gsplat::load_splat_splats(argv[1], sp, sh)) return 1;
    double sx=0, sy=0, sz=0, op=0;
    for (size_t i=0;i<sp.size();++i){ sx+=sp[i].x; sy+=sp[i].y; sz+=sp[i].z; op+=sp[i].opacity; }
    size_t n=sp.size();
    printf("  均值 pos=(%.2f, %.2f, %.2f)  opacity=%.3f\n", sx/n, sy/n, sz/n, op/n);
    printf("  第0个: (%.3f,%.3f,%.3f) op=%.3f sc=(%.4f,%.4f,%.4f) q=(%.3f,%.3f,%.3f,%.3f)\n",
        sp[0].x,sp[0].y,sp[0].z,sp[0].opacity,sp[0].scale0,sp[0].scale1,sp[0].scale2,
        sp[0].rot0,sp[0].rot1,sp[0].rot2,sp[0].rot3);
    printf("  第0个 sh.dc=(%.4f,%.4f,%.4f)\n", sh[0].dc[0],sh[0].dc[1],sh[0].dc[2]);
    return 0;
}
