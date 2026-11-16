"""Materiais PBR procedurais (Cycles) para os assets gerados no Blender.

Este módulo roda DENTRO de um processo com ``bpy``. Nenhum material usa textura
externa: todos são grafos de nós procedurais (ruídos, ondas, Voronoi), pensados
para parecer produto real numa foto — lã com nós e brilho de fibra, linho com
irregularidades de fio, madeira com anéis, poros e veios, metal escovado, etc.

Convenções:
- cores de entrada em sRGB (``"#RRGGBB"``, tupla 0..1 ou 0..255); convertidas para
  o espaço linear da cena;
- coordenadas de textura em METROS: ``"UV"`` (UVs geradas em metros pelos
  geradores de malha de ``procedural.py``, que acompanham a deformação — tapete
  enrolado, tecidos) ou ``"OBJECT"`` (espaço local do objeto, sem escala);
- abajures ficam APAGADOS: nenhum material emite luz, para não alterar a
  iluminação da foto original;
- materiais com os mesmos parâmetros são reaproveitados (cache pelo nome).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence

import bpy

ColorLike = str | Sequence[float]
RGB = tuple[float, float, float]

# --------------------------------------------------------------------------- cores


def srgb_to_linear(c: float) -> float:
    """Converte um canal sRGB (0..1) para linear."""
    c = min(max(float(c), 0.0), 1.0)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def parse_color(value: ColorLike | None, default: ColorLike) -> RGB:
    """Interpreta uma cor sRGB (hex, 0..1 ou 0..255) e devolve RGB linear."""
    v = default if value is None else value
    if isinstance(v, str):
        s = v.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            raise ValueError(f"Cor inválida: {value!r}")
        rgb = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    else:
        rgb = [float(x) for x in list(v)[:3]]
        if len(rgb) != 3:
            raise ValueError(f"Cor inválida: {value!r}")
        if max(rgb) > 1.0:
            rgb = [x / 255.0 for x in rgb]
    return tuple(srgb_to_linear(x) for x in rgb)  # type: ignore[return-value]


def _rgba(c: RGB, k: float = 1.0) -> tuple[float, float, float, float]:
    return (min(c[0] * k, 1.0), min(c[1] * k, 1.0), min(c[2] * k, 1.0), 1.0)


def _luma(c: RGB) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


# --------------------------------------------------------------------------- cache


def _cached(prefix: str, key: dict, build: Callable[[bpy.types.Material], None]) -> bpy.types.Material:
    """Devolve o material com esse nome+parâmetros, criando-o se ainda não existir."""
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]
    name = f"mob_{prefix}_{digest}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:  # versões antigas do Blender
        mat.use_nodes = True
    build(mat)
    mat["mobilia_material"] = prefix
    return mat


# --------------------------------------------------------------------------- construtor de grafos


class _G:
    """Construtor compacto de grafos de shader (Cycles)."""

    def __init__(self, mat: bpy.types.Material):
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.out.target = "ALL"
        self._tc: bpy.types.Node | None = None

    # -- básicos
    def new(self, idname: str, inputs: dict | None = None, **props) -> bpy.types.Node:
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (inputs or {}).items():
            self.set(n.inputs[k], v)
        return n

    def set(self, sock: bpy.types.NodeSocket, v) -> None:
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
            return
        dv = sock.default_value
        if isinstance(v, (int, float)) and hasattr(dv, "__len__"):
            n = len(dv)
            sock.default_value = (float(v),) * n if n != 4 else (float(v),) * 3 + (1.0,)
        elif hasattr(dv, "__len__") and len(dv) == 4 and len(v) == 3:
            sock.default_value = (*v, 1.0)
        else:
            sock.default_value = v

    @staticmethod
    def _sock(node: bpy.types.Node, ident: str, outputs: bool = False) -> bpy.types.NodeSocket:
        socks = node.outputs if outputs else node.inputs
        for s in socks:
            if s.identifier == ident:
                return s
        raise KeyError(ident)

    # -- coordenadas
    def coords(self, mode: str = "UV") -> bpy.types.NodeSocket:
        if self._tc is None:
            self._tc = self.new("ShaderNodeTexCoord")
        return self._tc.outputs["UV" if mode.upper() == "UV" else "Object"]

    # -- matemática
    def math(self, op: str, a, b=None, c=None, clamp: bool = False) -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.set(n.inputs[i], v)
        return n.outputs[0]

    def vmath(self, op: str, a, b=None, scale=None) -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeVectorMath", operation=op)
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if scale is not None:
            self.set(n.inputs["Scale"], scale)
        return n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]

    def sep(self, vec) -> bpy.types.Node:
        n = self.new("ShaderNodeSeparateXYZ")
        self.set(n.inputs[0], vec)
        return n

    def comb(self, x=0.0, y=0.0, z=0.0) -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.set(n.inputs[i], v)
        return n.outputs[0]

    def mapping(self, vec, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)) -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeMapping", vector_type="POINT")
        self.set(n.inputs["Vector"], vec)
        n.inputs["Location"].default_value = loc
        n.inputs["Rotation"].default_value = rot
        n.inputs["Scale"].default_value = scale
        return n.outputs[0]

    def map_range(self, v, fmin, fmax, tmin=0.0, tmax=1.0, clamp: bool = True,
                  interp: str = "LINEAR") -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeMapRange", data_type="FLOAT", interpolation_type=interp, clamp=clamp)
        self.set(n.inputs["Value"], v)
        self.set(n.inputs["From Min"], fmin)
        self.set(n.inputs["From Max"], fmax)
        self.set(n.inputs["To Min"], tmin)
        self.set(n.inputs["To Max"], tmax)
        return n.outputs["Result"]

    # -- texturas
    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0, lacunarity=2.0,
              ntype: str = "FBM") -> bpy.types.Node:
        n = self.new("ShaderNodeTexNoise", noise_dimensions="3D")
        n.noise_type = ntype
        self.set(n.inputs["Vector"], vec)
        self.set(n.inputs["Scale"], scale)
        self.set(n.inputs["Detail"], detail)
        self.set(n.inputs["Roughness"], rough)
        self.set(n.inputs["Lacunarity"], lacunarity)
        self.set(n.inputs["Distortion"], distortion)
        return n

    def voronoi(self, vec, scale, randomness=1.0, feature="F1", metric="EUCLIDEAN") -> bpy.types.Node:
        n = self.new("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature=feature, distance=metric)
        self.set(n.inputs["Vector"], vec)
        self.set(n.inputs["Scale"], scale)
        self.set(n.inputs["Randomness"], randomness)
        return n

    def white(self, vec) -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
        self.set(n.inputs["Vector"], vec)
        return n.outputs["Value"]

    def ramp(self, fac, stops: Sequence[tuple[float, Sequence[float]]], interp: str = "LINEAR") -> bpy.types.Node:
        n = self.new("ShaderNodeValToRGB")
        self.set(n.inputs[0], fac)
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        for i, (pos, col) in enumerate(stops):
            el = cr.elements[0] if i == 0 else cr.elements.new(pos)
            el.position = pos
            el.color = tuple(col) if len(col) == 4 else (*col, 1.0)
        return n

    # -- cores e shaders
    def mix(self, fac, a, b, blend: str = "MIX") -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeMix", data_type="RGBA", blend_type=blend, clamp_factor=True)
        self.set(self._sock(n, "Factor_Float"), fac)
        self.set(self._sock(n, "A_Color"), a)
        self.set(self._sock(n, "B_Color"), b)
        return self._sock(n, "Result_Color", outputs=True)

    def scale_color(self, col, k) -> bpy.types.NodeSocket:
        """Multiplica uma cor por um escalar (socket ou número)."""
        return self.mix(1.0, col, self.comb(k, k, k) if isinstance(k, bpy.types.NodeSocket) else (k, k, k),
                        blend="MULTIPLY")

    def bump(self, height, strength: float, distance: float, normal=None) -> bpy.types.NodeSocket:
        n = self.new("ShaderNodeBump")
        self.set(n.inputs["Height"], height)
        self.set(n.inputs["Strength"], strength)
        self.set(n.inputs["Distance"], distance)
        if normal is not None:
            self.set(n.inputs["Normal"], normal)
        return n.outputs["Normal"]

    def principled(self, **inp) -> bpy.types.Node:
        n = self.new("ShaderNodeBsdfPrincipled")
        n.distribution = "MULTI_GGX"
        names = {
            "base": "Base Color", "metallic": "Metallic", "rough": "Roughness", "ior": "IOR",
            "alpha": "Alpha", "normal": "Normal", "diffuse_rough": "Diffuse Roughness",
            "sss": "Subsurface Weight", "sss_radius": "Subsurface Radius", "sss_scale": "Subsurface Scale",
            "spec": "Specular IOR Level", "aniso": "Anisotropic", "aniso_rot": "Anisotropic Rotation",
            "tangent": "Tangent", "trans": "Transmission Weight", "coat": "Coat Weight",
            "coat_rough": "Coat Roughness", "sheen": "Sheen Weight", "sheen_rough": "Sheen Roughness",
            "sheen_tint": "Sheen Tint", "thin_wall": "Thin Wall",
        }
        for k, v in inp.items():
            if v is not None:
                self.set(n.inputs[names[k]], v)
        return n

    def output(self, shader_socket, displacement=None) -> None:
        self.nt.links.new(shader_socket, self.out.inputs["Surface"])
        if displacement is not None:
            self.nt.links.new(displacement, self.out.inputs["Displacement"])


def _seed_offset(seed: int) -> tuple[float, float, float]:
    h = hashlib.sha1(str(seed).encode()).digest()
    return tuple(37.0 * (h[i] / 255.0) - 18.0 for i in range(3))  # type: ignore[return-value]


def _variation(g: _G, P, scale: float, amount: float, detail: float = 2.0) -> bpy.types.NodeSocket:
    """Fator multiplicativo 1±amount a partir de um ruído suave."""
    n = g.noise(P, scale, detail=detail, rough=0.55)
    return g.map_range(n.outputs["Factor"], 0.3, 0.7, 1.0 - amount, 1.0 + amount)


# =========================================================================== tapete

RUG_PATTERNS = ("solid_border", "berber", "stripes", "persian")


def rug_wool(pattern: str, field: ColorLike | None, accent: ColorLike | None, length: float,
             depth: float, seed: int = 0, back: bool = False) -> bpy.types.Material:
    """Lã com nós (tapete). ``length``/``depth`` = extensão em metros das UVs (u, v).

    Padrões: ``solid_border`` (campo liso com faixa e filete), ``berber`` (losangos
    em linhas finas irregulares, estilo Beni Ourain), ``stripes`` (listras sutis
    simétricas) e ``persian`` (tradicional: bordas com rosetas, treliça no campo,
    medalhão central e cantoneiras). ``back=True`` gera o avesso (mesmo desenho, mais
    escuro e sem pelo).
    """
    if pattern not in RUG_PATTERNS:
        raise ValueError(f"Padrão de tapete desconhecido: {pattern!r}")
    defaults = {"solid_border": ("#d9d1c2", "#4b423b"), "berber": ("#e6dfd2", "#2f2a27"),
                "stripes": ("#cfc6b6", "#7a6a5a"), "persian": ("#7b3328", "#23324a")}
    fc = parse_color(field, defaults[pattern][0])
    ac = parse_color(accent, defaults[pattern][1])
    key = dict(p=pattern, f=fc, a=ac, L=round(length, 4), D=round(depth, 4), s=seed, b=back)

    shag = pattern == "berber"

    def build(mat):
        g = _G(mat)
        uv = g.coords("UV")
        knot = 0.0045                                          # passo do nó (m)
        q = g.vmath("ADD", g.vmath("SNAP", uv, (knot, knot, knot)), (knot / 2, knot / 2, 0.0))
        # irregularidade de feito à mão: ondulação larga (~1 cm) + tremor fino (~3 mm)
        w1 = g.noise(g.vmath("ADD", uv, _seed_offset(seed)), 1.6, detail=2.0)
        w2 = g.noise(g.vmath("ADD", uv, _seed_offset(seed + 9)), 7.0, detail=2.0)
        warp = g.vmath("ADD",
                       g.vmath("SCALE", g.vmath("SUBTRACT", w1.outputs["Color"], (0.5, 0.5, 0.5)), scale=0.03),
                       g.vmath("SCALE", g.vmath("SUBTRACT", w2.outputs["Color"], (0.5, 0.5, 0.5)), scale=0.012))
        qw = g.vmath("ADD", q, g.vmath("MULTIPLY", warp, (1.0, 1.0, 0.0)))
        s = g.sep(qw)
        u, v = s.outputs["X"], s.outputs["Y"]
        # tremor por nó na borda do desenho (contorno "pixelado" de tapete feito à mão)
        jit = g.math("MULTIPLY", g.math("SUBTRACT", g.white(g.vmath("ADD", q, _seed_offset(seed + 5))), 0.5), 0.004)
        if pattern == "persian":
            col, m = _persian(g, u, v, length, depth, fc, ac, jit)
        elif pattern == "stripes":
            acc, tone = _rug_pattern_mask(g, pattern, u, v, length, depth, seed, jit)
            col = g.mix(acc, g.mix(tone, _rgba(fc), _rgba(fc, 0.86)), _rgba(ac))
            m = acc
        else:
            m = _rug_pattern_mask(g, pattern, u, v, length, depth, seed, jit)
            col = g.mix(m, _rgba(fc), _rgba(ac))
        # tufos (Voronoi com domínio distorcido): cada tufo com tom e altura próprios
        dw = g.noise(g.vmath("ADD", uv, _seed_offset(seed + 8)), 160.0, detail=2.0)
        uvw = g.vmath("ADD", uv, g.vmath("SCALE", g.vmath("SUBTRACT", dw.outputs["Color"], (0.5, 0.5, 0.5)),
                                         scale=0.006 if shag else 0.003))
        tuft = g.voronoi(g.vmath("ADD", uvw, _seed_offset(seed + 6)), 70.0 if shag else 125.0, randomness=1.0)
        tr = g.white(tuft.outputs["Color"])
        dome = g.map_range(tuft.outputs["Distance"], 0.0, 0.75, 1.0, 0.0)
        # variação da lã: por tufo, mesclado (~4 cm), fio e "abrash" (faixas de tingimento)
        k_tuft = g.map_range(tr, 0.0, 1.0, 0.9, 1.08)
        mott = _variation(g, g.vmath("ADD", uv, _seed_offset(seed + 2)), 24.0, 0.06, detail=3.0)
        yarn = _variation(g, g.vmath("ADD", uv, _seed_offset(seed + 7)), 90.0, 0.04, detail=2.0)
        abr = _variation(g, g.mapping(g.vmath("ADD", uv, _seed_offset(seed + 3)), scale=(0.8, 10.0, 1.0)),
                         1.0, 0.04)
        k = g.math("MULTIPLY", g.math("MULTIPLY", k_tuft, mott), g.math("MULTIPLY", yarn, abr))
        col = g.scale_color(col, k)
        fib = g.noise(g.vmath("ADD", uv, _seed_offset(seed + 4)), 650.0, detail=3.0, rough=0.6)
        if back:
            # avesso: trama plana e mais escura (sem pelo)
            col = g.scale_color(col, 0.75)
            weave = g.math("MULTIPLY", g.math("SINE", g.math("MULTIPLY", u, 2 * math.pi / 0.003)),
                           g.math("SINE", g.math("MULTIPLY", v, 2 * math.pi / 0.003)))
            h = g.math("ADD", g.math("MULTIPLY", weave, 0.5), g.math("MULTIPLY", fib.outputs["Factor"], 0.3))
            nrm = g.bump(h, 0.25, 0.0008)
            bsdf = g.principled(base=col, rough=0.9, spec=0.3, sheen=0.25, sheen_rough=0.5,
                                diffuse_rough=0.6, normal=nrm)
        else:
            # sombra nas fendas entre tufos (oclusão fina que dá corpo ao pelo)
            crev = g.map_range(dome, 0.0, 0.3, 0.86 if shag else 0.9, 1.0)
            col = g.scale_color(col, crev)
            carve = 0.0 if pattern == "stripes" else g.math("MULTIPLY", m, -0.2 if pattern == "persian" else -0.25)
            h = g.math("ADD", g.math("ADD", g.math("MULTIPLY", dome, 0.65),
                                     g.math("MULTIPLY", fib.outputs["Factor"], 0.35)),
                       g.math("ADD", carve, g.math("MULTIPLY", tr, 0.25)))
            nrm = g.bump(h, 0.85 if shag else 0.5, 0.006 if shag else 0.003)
            bsdf = g.principled(base=col, rough=0.95, spec=0.2, sheen=0.85, sheen_rough=0.3,
                                sheen_tint=(1.0, 0.98, 0.95), diffuse_rough=0.85, normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached("rug_back" if back else "rug_wool", key, build)


def _rug_pattern_mask(g: _G, pattern: str, u, v, L: float, D: float, seed: int, jit):
    """Máscara 0..1 do desenho do tapete (1 = cor de destaque). ``jit`` = tremor por nó (m)."""
    if pattern == "solid_border":
        e = g.math("MINIMUM", g.math("MINIMUM", u, g.math("SUBTRACT", L, u)),
                   g.math("MINIMUM", v, g.math("SUBTRACT", D, v)))
        e = g.math("ADD", e, jit)
        band = max(0.05, min(0.09, 0.045 * min(L, D)))
        border = g.math("LESS_THAN", e, band)
        line_c, line_w = band + 0.055, 0.018
        line = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", e, line_c)), line_w / 2)
        return g.math("MAXIMUM", border, line)
    if pattern == "berber":
        nx = max(2, round(L / 0.5))
        ny = max(2, round(D / 0.6))
        sx, sy = L / nx, D / ny
        cu = g.math("DIVIDE", g.math("SUBTRACT", u, L / 2), sx)
        cv = g.math("DIVIDE", g.math("SUBTRACT", v, D / 2), sy)
        grad = math.sqrt(1.0 / sx ** 2 + 1.0 / sy ** 2)
        # espessura do traço varia ao longo da linha (lã tingida à mão)
        wv = g.noise(g.comb(u, v, float(seed % 89) + 0.5), 5.0, detail=2.0)
        lw = g.map_range(wv.outputs["Factor"], 0.3, 0.7, 0.011, 0.024)
        lines = None
        for sign in (1.0, -1.0):
            a = g.math("ADD", g.math("ADD", cu, g.math("MULTIPLY", cv, sign)), 0.5)
            d = g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", a), 0.5))
            dist = g.math("ADD", g.math("DIVIDE", d, grad), jit)
            ln = g.math("LESS_THAN", dist, g.math("MULTIPLY", lw, 0.5))
            lines = ln if lines is None else g.math("MAXIMUM", lines, ln)
        # falhas raras no traço (irregularidade artesanal)
        brk = g.noise(g.comb(u, v, float(seed % 97)), 3.0, detail=1.0)
        keep = g.math("GREATER_THAN", brk.outputs["Factor"], 0.24)
        # filete perto das cabeceiras
        e = g.math("ADD", g.math("MINIMUM", u, g.math("SUBTRACT", L, u)), jit)
        end1 = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", e, 0.075)), 0.008)
        return g.math("MAXIMUM", g.math("MULTIPLY", lines, keep), end1)
    # listras: desenho simétrico a partir do centro, ao longo do comprimento
    P = 0.42
    ua = g.math("ADD", g.math("ABSOLUTE", g.math("SUBTRACT", u, L / 2)), jit)
    x = g.math("MULTIPLY", g.math("FRACT", g.math("DIVIDE", ua, P)), P)
    acc = g.math("MAXIMUM",
                 g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", x, 0.23)), 0.006),
                 g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", x, 0.255)), 0.006))
    tone = g.math("MULTIPLY", g.math("GREATER_THAN", x, 0.30), g.math("LESS_THAN", x, 0.39))
    return acc, tone


def _persian(g: _G, u, v, L: float, D: float, fc: RGB, ac: RGB, jit):
    """Desenho tradicional (persa simplificado): guarda externa escura, borda principal com
    ramo ondulado e rosetas miúdas, guardas claras, campo com treliça fina, medalhão
    central em camadas e cantoneiras. Tons suavizados (aspecto de lã tingida, levemente
    envelhecida). Devolve (cor, máscara de contornos)."""
    mixc = lambda a, b, t: tuple(x * (1 - t) + y * t for x, y in zip(a, b))  # noqa: E731
    ivory = mixc((0.72, 0.64, 0.5), fc, 0.12)
    dark = mixc(tuple(0.32 * c for c in fc), ac, 0.35)
    soft_ac = mixc(ac, fc, 0.15)
    m = lambda op, a, b=None: g.math(op, a, b)  # noqa: E731
    lt = lambda a, b: m("LESS_THAN", a, b)      # noqa: E731
    band = lambda x, a, b: m("MULTIPLY", m("GREATER_THAN", x, a), lt(x, b))  # noqa: E731
    du = m("MINIMUM", u, m("SUBTRACT", L, u))
    dv = m("MINIMUM", v, m("SUBTRACT", D, v))
    e = m("ADD", m("MINIMUM", du, dv), jit)
    g0 = 0.022
    bm = float(min(max(0.075 * min(L, D), 0.08), 0.17))
    g1 = g0 + bm
    g2 = g1 + 0.03
    # --- borda principal: ramo ondulado com rosetas miúdas e folhas, ao longo da borda
    along = g.mix(lt(du, dv), g.comb(u, u, u), g.comb(v, v, v))
    s_al = g.sep(along).outputs["X"]
    per = bm * 0.85
    xl = m("MULTIPLY", m("SUBTRACT", m("FRACT", m("DIVIDE", s_al, per)), 0.5), per)
    wave = m("MULTIPLY", g.math("SINE", m("MULTIPLY", s_al, 2 * math.pi / per)), 0.22 * bm)
    yl = m("SUBTRACT", e, g0 + bm / 2)
    yw = m("SUBTRACT", yl, wave)
    r = m("SQRT", m("ADD", m("MULTIPLY", xl, xl), m("MULTIPLY", yl, yl)))
    vine = lt(m("ABSOLUTE", yw), 0.005)
    ros = band(r, 0.09 * bm, 0.2 * bm)
    ros_c = lt(r, 0.05 * bm)
    # folhas: pequenas manchas alternadas dos dois lados do ramo
    xl2 = m("MULTIPLY", m("SUBTRACT", m("FRACT", m("ADD", m("DIVIDE", s_al, per), 0.5)), 0.5), per)
    leaf = lt(m("ADD", m("ABSOLUTE", m("MULTIPLY", xl2, 1.6)), m("ABSOLUTE", m("ADD", yl, m("MULTIPLY", wave, -1.0)))),
              0.11 * bm)
    border = g.mix(vine, _rgba(soft_ac), _rgba(mixc(fc, ivory, 0.25)))
    border = g.mix(m("MULTIPLY", leaf, 0.85), border, _rgba(mixc(fc, soft_ac, 0.3)))
    border = g.mix(ros, border, _rgba(mixc(ivory, fc, 0.2)))
    border = g.mix(ros_c, border, _rgba(fc))
    guard_out = band(e, g0 - 0.008, g0)
    guard_in = band(e, g1, g2)
    guard_line = lt(m("ABSOLUTE", m("SUBTRACT", e, (g1 + g2) / 2)), 0.0035)
    border = g.mix(guard_in, border, _rgba(mixc(ivory, fc, 0.15)))
    border = g.mix(m("MULTIPLY", guard_in, guard_line), border, _rgba(dark))
    border = g.mix(guard_out, border, _rgba(mixc(ivory, fc, 0.3)))
    border = g.mix(lt(e, g0 - 0.008), border, _rgba(dark))
    # --- campo: treliça fina e discreta
    lat = 0.1
    xf = m("ABSOLUTE", m("SUBTRACT", m("FRACT", m("DIVIDE", u, lat)), 0.5))
    yf = m("ABSOLUTE", m("SUBTRACT", m("FRACT", m("DIVIDE", v, lat)), 0.5))
    dmd = m("ADD", xf, yf)
    small = band(dmd, 0.2, 0.26)
    dot = lt(dmd, 0.06)
    field = g.mix(m("MULTIPLY", small, 0.45), _rgba(fc), _rgba(soft_ac))
    field = g.mix(m("MULTIPLY", dot, 0.5), field, _rgba(ivory))
    # --- cantoneiras (a partir dos cantos do campo)
    fu = m("SUBTRACT", du, g2)
    fv = m("SUBTRACT", dv, g2)
    dc = m("ADD", m("DIVIDE", fu, 0.15 * L), m("DIVIDE", fv, 0.19 * D))
    corner = g.mix(m("MULTIPLY", small, 0.6), _rgba(soft_ac), _rgba(ivory))
    field = g.mix(lt(dc, 1.0), field, corner)
    field = g.mix(band(dc, 0.9, 1.0), field, _rgba(ivory))
    # --- medalhão central em camadas
    dn = m("ADD", m("DIVIDE", m("ABSOLUTE", m("SUBTRACT", u, L / 2)), 0.19 * L),
           m("DIVIDE", m("ABSOLUTE", m("SUBTRACT", v, D / 2)), 0.28 * D))
    dn = m("ADD", dn, m("MULTIPLY", jit, 4.0))
    med = g.mix(m("MULTIPLY", small, 0.7), _rgba(soft_ac), _rgba(ivory))
    med = g.mix(lt(dn, 0.52), med, _rgba(mixc(fc, ivory, 0.08)))
    med = g.mix(m("MULTIPLY", band(dn, 0.2, 0.52), small), med, _rgba(soft_ac))
    med = g.mix(band(dn, 0.46, 0.52), med, _rgba(ivory))
    med = g.mix(lt(dn, 0.14), med, _rgba(dark))
    field = g.mix(lt(dn, 1.0), field, med)
    field = g.mix(band(dn, 0.93, 1.0), field, _rgba(ivory))
    col = g.mix(lt(e, g2), field, border)
    outline = m("MAXIMUM", m("MAXIMUM", band(dn, 0.93, 1.0), band(dc, 0.9, 1.0)), m("MAXIMUM", guard_line, ros))
    return col, outline


def rug_fringe(color: ColorLike | None = None) -> bpy.types.Material:
    """Franja de algodão cru."""
    c = parse_color(color, "#e8e0cf")

    def build(mat):
        g = _G(mat)
        uv = g.coords("UV")
        n = g.noise(g.mapping(uv, scale=(30.0, 400.0, 1.0)), 1.0, detail=3.0)
        col = g.scale_color(_rgba(c), g.map_range(n.outputs["Factor"], 0.3, 0.7, 0.88, 1.06))
        nrm = g.bump(n.outputs["Factor"], 0.3, 0.0005)
        bsdf = g.principled(base=col, rough=0.85, spec=0.3, sheen=0.6, sheen_rough=0.35, normal=nrm,
                            diffuse_rough=0.5)
        g.output(bsdf.outputs[0])

    return _cached("fringe", dict(c=c), build)


# =========================================================================== tecidos

FABRICS = ("linen", "velvet", "boucle", "cotton")


def fabric(color: ColorLike | None, kind: str = "linen", coords: str = "UV", seed: int = 0,
           translucency: float = 0.0, sheer: bool = False) -> bpy.types.Material:
    """Tecido de estofado/cortina: linho (fios irregulares), veludo, bouclê ou algodão.

    ``translucency`` (0..1) mistura um BSDF translúcido (cúpulas, cortinas finas);
    ``sheer=True`` deixa o tecido semitransparente (voal), para cortinas leves.
    """
    if kind not in FABRICS:
        raise ValueError(f"Tecido desconhecido: {kind!r}")
    c = parse_color(color, "#b9b1a4")
    key = dict(c=c, k=kind, co=coords, s=seed, t=round(translucency, 3), sh=sheer)

    def build(mat):
        g = _G(mat)
        P = g.vmath("ADD", g.coords(coords), _seed_offset(seed))
        if kind == "boucle":
            vor = g.voronoi(P, 260.0, randomness=1.0)
            loops = g.map_range(vor.outputs["Distance"], 0.0, 0.5, 1.0, 0.0)
            fuzz = g.noise(P, 900.0, detail=2.0)
            h = g.math("ADD", g.math("POWER", loops, 0.6), g.math("MULTIPLY", fuzz.outputs["Factor"], 0.4))
            var = g.map_range(g.white(vor.outputs["Color"]), 0.0, 1.0, 0.9, 1.07)
            col = g.scale_color(_rgba(c), g.math("MULTIPLY", var, _variation(g, P, 4.0, 0.03)))
            col = g.mix(g.math("MULTIPLY", g.math("SUBTRACT", 1.0, loops), 0.35), col, _rgba(c, 0.62))
            nrm = g.bump(h, 0.6, 0.0025)
            bsdf = g.principled(base=col, rough=0.95, spec=0.25, sheen=0.55, sheen_rough=0.4,
                                diffuse_rough=0.8, normal=nrm)
        elif kind == "velvet":
            crush = g.noise(g.mapping(P, scale=(3.0, 7.0, 3.0)), 1.0, detail=3.0, rough=0.6)
            fine = g.noise(P, 600.0, detail=1.0)
            col = g.scale_color(_rgba(c), g.map_range(crush.outputs["Factor"], 0.25, 0.75, 0.88, 1.08))
            nrm = g.bump(g.math("ADD", crush.outputs["Factor"], g.math("MULTIPLY", fine.outputs["Factor"], 0.2)),
                         0.12, 0.004)
            sheen_tint = _rgba(tuple(min(1.0, 0.55 + 0.6 * x) for x in c))
            bsdf = g.principled(base=col, rough=0.7, spec=0.2, sheen=1.0, sheen_rough=0.22,
                                sheen_tint=sheen_tint, diffuse_rough=0.6, normal=nrm)
        else:
            pitch = 0.0011 if kind == "linen" else 0.0006
            sl_u = g.noise(g.mapping(P, scale=(9.0, 260.0, 9.0)), 1.0, detail=2.0, rough=0.6)
            sl_v = g.noise(g.mapping(P, scale=(260.0, 9.0, 260.0)), 1.0, detail=2.0, rough=0.6)
            sp = g.sep(P)
            weave = g.math("MULTIPLY", g.math("SINE", g.math("MULTIPLY", sp.outputs["X"], 2 * math.pi / pitch)),
                           g.math("SINE", g.math("MULTIPLY", sp.outputs["Y"], 2 * math.pi / pitch)))
            slub_amt = 0.09 if kind == "linen" else 0.03
            k = g.math("MULTIPLY",
                       g.map_range(sl_u.outputs["Factor"], 0.3, 0.7, 1.0 - slub_amt, 1.0 + slub_amt),
                       g.map_range(sl_v.outputs["Factor"], 0.3, 0.7, 1.0 - slub_amt * 0.7, 1.0 + slub_amt * 0.7))
            k = g.math("MULTIPLY", k, _variation(g, P, 2.5, 0.025))
            col = g.scale_color(_rgba(c), k)
            h = g.math("ADD", g.math("MULTIPLY", weave, 0.25),
                       g.math("ADD", sl_u.outputs["Factor"], sl_v.outputs["Factor"]))
            nrm = g.bump(h, 0.22 if kind == "linen" else 0.12, 0.0012)
            bsdf = g.principled(base=col, rough=0.84 if kind == "linen" else 0.76, spec=0.32,
                                sheen=0.45 if kind == "linen" else 0.5, sheen_rough=0.4,
                                diffuse_rough=0.55, normal=nrm)
        shader = bsdf.outputs[0]
        if translucency > 0.0:
            tr = g.new("ShaderNodeBsdfTranslucent", {"Color": _rgba(c, 0.9)})
            mixs = g.new("ShaderNodeMixShader", {"Factor": translucency})
            g.nt.links.new(shader, mixs.inputs[1])
            g.nt.links.new(tr.outputs[0], mixs.inputs[2])
            shader = mixs.outputs[0]
        if sheer:
            tp = g.new("ShaderNodeBsdfTransparent", {"Color": (1.0, 1.0, 1.0, 1.0)})
            m2 = g.new("ShaderNodeMixShader", {"Factor": 0.55})
            g.nt.links.new(shader, m2.inputs[1])
            g.nt.links.new(tp.outputs[0], m2.inputs[2])
            shader = m2.outputs[0]
        g.output(shader)

    return _cached(f"fabric_{kind}", key, build)


def lamp_shade(color: ColorLike | None = None, seed: int = 0) -> bpy.types.Material:
    """Cúpula de linho levemente translúcida (lâmpada apagada: sem emissão)."""
    return fabric(color if color is not None else "#efe9dd", "linen", coords="UV", seed=seed, translucency=0.3)


def bed_linen(color: ColorLike | None = None, wrinkles: float = 1.0, seed: int = 0,
              coords: str = "UV") -> bpy.types.Material:
    """Roupa de cama de algodão: brilho suave de fibra e amassados finos no relevo."""
    c = parse_color(color, "#f1eee8")
    key = dict(c=c, w=round(wrinkles, 3), s=seed, co=coords)

    def build(mat):
        g = _G(mat)
        P = g.vmath("ADD", g.coords(coords), _seed_offset(seed))
        crease = []
        for rot, sc in ((0.0, (5.0, 16.0, 5.0)), (1.1, (4.0, 12.0, 4.0)), (-0.7, (9.0, 26.0, 9.0))):
            n = g.noise(g.mapping(P, rot=(0.0, 0.0, rot), scale=sc), 1.0, detail=2.0, rough=0.45)
            ridge = g.math("SUBTRACT", 1.0, g.math("ABSOLUTE", g.math("SUBTRACT", g.math("MULTIPLY", n.outputs["Factor"], 2.0), 1.0)))
            crease.append(g.math("POWER", ridge, 4.0))
        cr = g.math("ADD", g.math("MULTIPLY", crease[0], 0.5),
                    g.math("ADD", g.math("MULTIPLY", crease[1], 0.35), g.math("MULTIPLY", crease[2], 0.25)))
        soft = g.noise(P, 2.5, detail=1.0)
        weave = g.noise(P, 2200.0, detail=1.0)
        h = g.math("ADD", g.math("MULTIPLY", cr, wrinkles),
                   g.math("ADD", g.math("MULTIPLY", soft.outputs["Factor"], 0.6),
                          g.math("MULTIPLY", weave.outputs["Factor"], 0.08)))
        nrm = g.bump(h, 0.4, 0.006)
        k = g.math("MULTIPLY", g.map_range(soft.outputs["Factor"], 0.3, 0.7, 0.975, 1.02),
                   g.math("SUBTRACT", 1.0, g.math("MULTIPLY", cr, 0.04 * wrinkles)))
        col = g.scale_color(_rgba(c), k)
        bsdf = g.principled(base=col, rough=0.74, spec=0.33, sheen=0.5, sheen_rough=0.3, diffuse_rough=0.5,
                            sss=0.03, sss_radius=(0.004, 0.003, 0.002), normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached("linen_bed", key, build)


def knit(color: ColorLike | None = None, accent: ColorLike | None = None, pattern: str = "solid",
         seed: int = 0, width: float = 1.2) -> bpy.types.Material:
    """Malha de tricô grosso (manta): colunas de pontos em "V", sombra entre os pontos,
    fiapos com brilho de fibra e, opcionalmente, listras perto das pontas franjadas.

    UVs em metros: u ao longo da largura da manta (0..``width``), v ao longo do tecido.
    """
    c = parse_color(color, "#c9b49b")
    a = parse_color(accent, "#f0e9dc")
    key = dict(c=c, a=a, p=pattern, s=seed, w=round(width, 3))

    def build(mat):
        g = _G(mat)
        uv = g.coords("UV")
        s = g.sep(uv)
        u, v = s.outputs["X"], s.outputs["Y"]
        cw, rh = 0.016, 0.012                       # largura da coluna e altura da carreira (m)
        cu = g.math("SUBTRACT", g.math("FRACT", g.math("DIVIDE", u, cw)), 0.5)
        au = g.math("ABSOLUTE", cu)
        # cada coluna tem duas "pernas" inclinadas (o V do ponto de meia)
        cv = g.math("FRACT", g.math("DIVIDE", g.math("ADD", v, g.math("MULTIPLY", au, rh * 1.1)), rh))
        leg = g.math("SUBTRACT", 1.0, g.math("POWER", g.math("ABSOLUTE", g.math("SUBTRACT", g.math("MULTIPLY", au, 4.0), 1.0)), 2.0))
        loop = g.math("ADD", 0.45, g.math("MULTIPLY", g.math("SINE", g.math("MULTIPLY", cv, math.pi)), 0.55))
        stitch = g.math("MULTIPLY", g.math("MAXIMUM", leg, 0.0), loop)
        fuzz = g.noise(g.vmath("ADD", uv, _seed_offset(seed)), 700.0, detail=2.0)
        h = g.math("ADD", stitch, g.math("MULTIPLY", fuzz.outputs["Factor"], 0.3))
        nrm = g.bump(h, 0.9, 0.006)
        base = _rgba(c)
        if pattern == "stripes":
            # três listras finas perto de cada ponta franjada (paralelas à franja)
            e = g.math("MINIMUM", u, g.math("SUBTRACT", width, u))
            st = None
            for c0 in (0.07, 0.1, 0.13):
                x = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", e, c0)), 0.009)
                st = x if st is None else g.math("MAXIMUM", st, x)
            col = g.mix(st, base, _rgba(a))
        else:
            col = base
        k = g.math("MULTIPLY", g.map_range(stitch, 0.0, 1.0, 0.68, 1.06),
                   _variation(g, g.vmath("ADD", uv, _seed_offset(seed + 1)), 3.0, 0.04))
        col = g.scale_color(col, k)
        bsdf = g.principled(base=col, rough=0.93, spec=0.22, sheen=1.0, sheen_rough=0.5,
                            diffuse_rough=0.85, normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached("knit", key, build)


# =========================================================================== metal, madeira, pintura

METAL_FINISHES = ("black", "graphite", "brass", "chrome", "matte_black", "white")


def metal_finish_for_color(color: ColorLike) -> str:
    """Acabamento de metal mais próximo de uma cor da paleta (ex.: "#B08D57" -> "brass")."""
    rgb = parse_color(color, "#1f1f21")
    lum = _luma(rgb)
    hi_, lo_ = max(rgb), min(rgb)
    sat = (hi_ - lo_) / hi_ if hi_ > 1e-6 else 0.0
    if sat > 0.3 and rgb[0] > rgb[2] and lum > 0.06:
        return "brass"
    if lum < 0.03:
        return "black"
    if lum < 0.2:
        return "graphite"
    if lum > 0.5 and sat < 0.15:
        return "white"
    return "chrome"


def metal(finish: str = "black", brush_axis: str = "Z", seed: int = 0) -> bpy.types.Material:
    """Metal escovado (preto, grafite, latão, cromo) ou pintura eletrostática (preta fosca
    ou branca acetinada).

    ``brush_axis``: direção do escovado no espaço do objeto (``X``, ``Y``, ``Z``) ou
    ``RADIAL`` (escovado circular, como em bases torneadas).
    """
    if finish not in METAL_FINISHES:
        raise ValueError(f"Acabamento de metal desconhecido: {finish!r}")
    key = dict(f=finish, b=brush_axis, s=seed)

    def build(mat):
        g = _G(mat)
        P = g.vmath("ADD", g.coords("OBJECT"), _seed_offset(seed))
        if finish in ("matte_black", "white"):  # pintura eletrostática ("casca de laranja" fina)
            peel = g.noise(P, 1400.0, detail=2.0)
            var = _variation(g, P, 6.0, 0.06 if finish == "matte_black" else 0.015)
            nrm = g.bump(peel.outputs["Factor"], 0.06, 0.0005)
            base = (0.014, 0.014, 0.015, 1.0) if finish == "matte_black" else (0.78, 0.77, 0.74, 1.0)
            bsdf = g.principled(base=g.scale_color(base, var), rough=0.42 if finish == "matte_black" else 0.36,
                                spec=0.45, coat=0.15, coat_rough=0.35, normal=nrm)
            g.output(bsdf.outputs[0])
            return
        if brush_axis == "RADIAL":
            sp = g.sep(P)
            r = g.vmath("LENGTH", g.comb(sp.outputs["X"], sp.outputs["Y"], 0.0))
            bv = g.comb(g.math("MULTIPLY", r, 1.0), 0.0, 0.0)
            streak = g.noise(g.mapping(bv, scale=(2500.0, 1.0, 1.0)), 1.0, detail=3.0)
        else:
            sc = {"X": (3.0, 1800.0, 1800.0), "Y": (1800.0, 3.0, 1800.0), "Z": (1800.0, 1800.0, 3.0)}[brush_axis]
            streak = g.noise(g.mapping(P, scale=sc), 1.0, detail=3.0)
        sf = streak.outputs["Factor"]
        tang = g.new("ShaderNodeTangent", direction_type="RADIAL",
                     axis="Z" if brush_axis in ("Z", "RADIAL") else brush_axis)
        if finish == "brass":
            base, rough, aniso = (0.78, 0.56, 0.27), 0.26, 0.55
        elif finish == "chrome":
            base, rough, aniso = (0.62, 0.62, 0.63), 0.08, 0.2
        elif finish == "graphite":  # ferro grafite escovado
            base, rough, aniso = (0.11, 0.108, 0.105), 0.32, 0.5
        else:  # preto escovado (aço escurecido)
            base, rough, aniso = (0.035, 0.034, 0.033), 0.3, 0.5
        smudge = g.noise(P, 7.0, detail=3.0)
        rgh = g.math("ADD", g.math("ADD", rough, g.math("MULTIPLY", g.math("SUBTRACT", sf, 0.5), 0.12)),
                     g.math("MULTIPLY", g.math("SUBTRACT", smudge.outputs["Factor"], 0.5), 0.06))
        col = g.scale_color(_rgba(base), g.map_range(smudge.outputs["Factor"], 0.3, 0.7, 0.95, 1.04))
        nrm = g.bump(sf, 0.03, 0.0003)
        bsdf = g.principled(base=col, metallic=1.0, rough=rgh, aniso=aniso, tangent=tang.outputs[0],
                            normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached(f"metal_{finish}", key, build)


WOODS = ("oak", "walnut")


def wood(species: str = "oak", grain_axis: str = "X", seed: int = 0,
         tint: ColorLike | None = None) -> bpy.types.Material:
    """Madeira procedural (carvalho ou nogueira) com anéis, veios, poros e acabamento acetinado.

    ``grain_axis`` = eixo do veio no espaço do objeto (``X``, ``Y`` ou ``Z``).
    ``tint`` (opcional) = tom médio desejado (cor da paleta, ex.: "#A47A50"); o desenho do
    veio continua o da espécie.
    """
    if species not in WOODS:
        raise ValueError(f"Madeira desconhecida: {species!r}")
    tint_lin = parse_color(tint, "#000000") if tint is not None else None
    key = dict(sp=species, ax=grain_axis, s=seed, t=tint_lin)

    def build(mat):
        g = _G(mat)
        P0 = g.vmath("ADD", g.coords("OBJECT"), _seed_offset(seed))
        s0 = g.sep(P0)
        X, Y, Z = s0.outputs["X"], s0.outputs["Y"], s0.outputs["Z"]
        a, b, c = {"X": (X, Y, Z), "Y": (Y, X, Z), "Z": (Z, X, Y)}[grain_axis]
        P = g.comb(a, b, c)  # veio ao longo de x'
        # tronco fora da peça, levemente inclinado: desenho "catedral" nas faces
        dist = g.noise(g.mapping(P, scale=(1.2, 5.0, 5.0)), 1.0, detail=3.0, rough=0.55)
        dy = g.math("ADD", g.math("SUBTRACT", b, 0.23), g.math("MULTIPLY", a, 0.035))
        dz = g.math("ADD", g.math("ADD", c, 0.31), g.math("MULTIPLY", a, 0.02))
        r = g.math("ADD", g.math("SQRT", g.math("ADD", g.math("MULTIPLY", dy, dy), g.math("MULTIPLY", dz, dz))),
                   g.math("MULTIPLY", g.math("SUBTRACT", dist.outputs["Factor"], 0.5), 0.016))
        density = 190.0 if species == "oak" else 150.0
        ring = g.math("FRACT", g.math("MULTIPLY", r, density))
        late = g.ramp(ring, [(0.0, (0.0, 0.0, 0.0)), (0.62, (0.12, 0.12, 0.12)), (0.86, (1.0, 1.0, 1.0)),
                             (1.0, (0.25, 0.25, 0.25))]).outputs["Color"]
        late_f = g.sep(late).outputs["X"]
        streak = g.noise(g.mapping(P, scale=(2.5, 160.0, 160.0)), 1.0, detail=4.0, rough=0.6)
        pores = g.noise(g.mapping(P, scale=(25.0, 1600.0, 1600.0)), 1.0, detail=1.0)
        pore = g.map_range(pores.outputs["Factor"], 0.6, 0.72, 0.0, 1.0)
        if species == "oak":
            light, dark, pore_c = (0.56, 0.37, 0.2), (0.36, 0.21, 0.1), (0.2, 0.12, 0.06)
            pore_amt = 0.5
        else:
            light, dark, pore_c = (0.2, 0.11, 0.06), (0.075, 0.04, 0.022), (0.04, 0.022, 0.012)
            pore_amt = 0.35
        if tint_lin is not None:  # mantém o contraste do veio, centrado no tom pedido
            light = tuple(min(1.0, c * 1.2) for c in tint_lin)
            dark = tuple(c * 0.66 for c in tint_lin)
            pore_c = tuple(c * 0.4 for c in tint_lin)
        t = g.math("ADD", g.math("MULTIPLY", late_f, 0.55), g.math("MULTIPLY", streak.outputs["Factor"], 0.45),
                   clamp=True)
        col = g.mix(t, _rgba(light), _rgba(dark))
        col = g.mix(g.math("MULTIPLY", pore, pore_amt), col, _rgba(pore_c))
        broad = _variation(g, P, 1.4, 0.06)
        col = g.scale_color(col, broad)
        rgh = g.math("ADD", g.math("ADD", 0.46, g.math("MULTIPLY", pore, 0.16)),
                     g.math("MULTIPLY", g.math("SUBTRACT", streak.outputs["Factor"], 0.5), 0.1))
        h = g.math("SUBTRACT", g.math("MULTIPLY", late_f, 0.15), g.math("MULTIPLY", pore, 0.6))
        nrm = g.bump(h, 0.12, 0.0006)
        bsdf = g.principled(base=col, rough=rgh, spec=0.5, coat=0.12, coat_rough=0.3, diffuse_rough=0.2,
                            normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached(f"wood_{species}", key, build)


def painted(color: ColorLike | None, finish: str = "satin", seed: int = 0) -> bpy.types.Material:
    """Superfície pintada (fosco, acetinado ou brilhante) com leve "casca de laranja"."""
    c = parse_color(color, "#ece8e0")
    rough = {"matte": 0.62, "satin": 0.38, "gloss": 0.14}[finish]
    key = dict(c=c, f=finish, s=seed)

    def build(mat):
        g = _G(mat)
        P = g.vmath("ADD", g.coords("OBJECT"), _seed_offset(seed))
        peel = g.noise(P, 1200.0, detail=2.0)
        nrm = g.bump(peel.outputs["Factor"], 0.04, 0.0004)
        col = g.scale_color(_rgba(c), _variation(g, P, 5.0, 0.015))
        bsdf = g.principled(base=col, rough=rough, spec=0.5, coat=0.2 if finish == "gloss" else 0.0,
                            coat_rough=0.06, normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached("painted", key, build)


def ceramic(color: ColorLike | None, finish: str = "gloss", seed: int = 0) -> bpy.types.Material:
    """Cerâmica vitrificada com variação de esmalte e pintas (base de abajur)."""
    c = parse_color(color, "#cdbfae")
    key = dict(c=c, f=finish, s=seed)

    def build(mat):
        g = _G(mat)
        P = g.vmath("ADD", g.coords("OBJECT"), _seed_offset(seed))
        pool = g.noise(P, 9.0, detail=4.0, rough=0.6)
        speck = g.voronoi(P, 260.0, randomness=1.0)
        dots = g.math("LESS_THAN", speck.outputs["Distance"], 0.06)
        col = g.scale_color(_rgba(c), g.map_range(pool.outputs["Factor"], 0.25, 0.75, 0.86, 1.08))
        col = g.mix(g.math("MULTIPLY", dots, 0.8), col, _rgba(c, 0.35))
        rough = 0.07 if finish == "gloss" else 0.32
        nrm = g.bump(pool.outputs["Factor"], 0.05, 0.002)
        bsdf = g.principled(base=col, rough=rough, spec=0.5, coat=0.6 if finish == "gloss" else 0.2,
                            coat_rough=0.04, sss=0.04, sss_radius=(0.003, 0.002, 0.002), normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached("ceramic", key, build)


def opal_bulb() -> bpy.types.Material:
    """Lâmpada leitosa APAGADA: vidro opalino branco (sem emissão e sem transmissão
    refrativa — não confundir com vidro transparente)."""

    def build(mat):
        g = _G(mat)
        bsdf = g.principled(base=(0.86, 0.86, 0.84, 1.0), rough=0.28, spec=0.5, coat=0.6, coat_rough=0.05,
                            sss=0.35, sss_radius=(0.01, 0.01, 0.01), sss_scale=0.01)
        g.output(bsdf.outputs[0])

    return _cached("bulb_opal", {}, build)


def paper_edges(color: ColorLike | None = None, seed: int = 0) -> bpy.types.Material:
    """Corte das páginas de um livro: papel fosco com as estrias finas das folhas (paralelas ao
    tampo, ou seja, variando só em Z no referencial da peça)."""
    c = parse_color(color, "#ede6d6")
    key = dict(c=c, s=seed)

    def build(mat):
        g = _G(mat)
        P = g.vmath("ADD", g.coords("OBJECT"), _seed_offset(seed))
        leaves = g.noise(g.mapping(P, scale=(30.0, 30.0, 2600.0)), 1.0, detail=1.0)
        col = g.scale_color(_rgba(c), g.map_range(leaves.outputs["Factor"], 0.3, 0.7, 0.9, 1.03))
        col = g.scale_color(col, _variation(g, P, 6.0, 0.03))
        nrm = g.bump(leaves.outputs["Factor"], 0.2, 0.0003)
        bsdf = g.principled(base=col, rough=0.88, spec=0.3, diffuse_rough=0.6, normal=nrm)
        g.output(bsdf.outputs[0])

    return _cached("paper", key, build)


def felt(color: ColorLike | None = None) -> bpy.types.Material:
    """Feltro (sola de bases)."""
    c = parse_color(color, "#3b3a38")

    def build(mat):
        g = _G(mat)
        P = g.coords("OBJECT")
        n = g.noise(P, 900.0, detail=2.0)
        bsdf = g.principled(base=_rgba(c), rough=1.0, spec=0.2, sheen=0.5,
                            normal=g.bump(n.outputs["Factor"], 0.3, 0.0005))
        g.output(bsdf.outputs[0])

    return _cached("felt", dict(c=c), build)
