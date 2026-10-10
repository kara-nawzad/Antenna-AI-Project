"""
Interactive Plotly charts for the antenna synthesis app.

Replaces the static matplotlib PNGs in the forward-design section. The numbers come
from the engine unchanged -- this module only renders them. Plotly figures are
vector/SVG in the browser, so they cost far less bandwidth than the base64 raster
PNGs Streamlit was pushing on every rerun, and they give hover readouts of the exact
S11/VSWR at any frequency, which is what an RF engineer actually wants from a plot.

Note on Smith charts: a Smith chart needs complex S11 (magnitude AND phase). The CST
dataset stores magnitude in dB only, so a true Smith chart cannot be drawn from it
without fabricating phase. The VSWR trace below carries the same matching information
that is actually supported by the data.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared dark palette, matched to the app's existing CSS.
BG = "rgba(0,0,0,0)"
PAPER = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.09)"
AXIS = "#a8b0bd"
TEXT = "#eceff3"
S11_COLOR = "#4da3ff"
VSWR_COLOR = "#3fd6b0"
ACCENT = "#ff9838"
DANGER = "#ff5470"
GOOD = "#57d977"

SERIES = ["#4da3ff", "#ff9838", "#3fd6b0", "#c58bff", "#ffd166", "#ff5470"]


def _style(fig, height=430):
    fig.update_layout(
        template=None,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", size=12, color=TEXT),
        height=height,
        margin=dict(l=58, r=24, t=54, b=48),
        hoverlabel=dict(bgcolor="#1b1f27", font_size=12, bordercolor="#3a4150"),
        legend=dict(orientation="h", yanchor="bottom", y=1.005, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


def _axes(fig, **kw):
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#3a4150",
                     tickfont=dict(color=AXIS, size=11), **kw)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#3a4150",
                     tickfont=dict(color=AXIS, size=11))
    return fig


def s11_vswr_fig(f, c, v, fr=None, bw=None, title="Predicted Response"):
    """S11 and VSWR on a shared frequency axis, with the -10 dB band shaded."""
    f = np.asarray(f, dtype=float)
    c = np.asarray(c, dtype=float)
    v = np.asarray(v, dtype=float)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # -10 dB matching band
    in_band = c <= -10.0
    if np.any(in_band):
        fig.add_trace(go.Scatter(
            x=f, y=np.where(in_band, c, np.nan),
            mode="lines", line=dict(color=GOOD, width=0),
            fill="tozeroy", fillcolor="rgba(87,217,119,0.10)",
            name="matched (&le; -10 dB)", hoverinfo="skip", showlegend=True,
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=f, y=c, name="S11 (dB)", mode="lines",
        line=dict(color=S11_COLOR, width=2.6),
        hovertemplate="%{x:.3f} GHz<br>S11 = %{y:.2f} dB<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=f, y=v, name="VSWR", mode="lines",
        line=dict(color=VSWR_COLOR, width=2.2, dash="dot"),
        hovertemplate="%{x:.3f} GHz<br>VSWR = %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=-10, line=dict(color=DANGER, width=1.4, dash="dash"),
                  annotation_text="-10 dB", annotation_position="bottom right",
                  annotation_font=dict(color=DANGER, size=11), secondary_y=False)
    fig.add_hline(y=2, line=dict(color=DANGER, width=1.2, dash="dash"),
                  annotation_text="VSWR 2", annotation_position="top right",
                  annotation_font=dict(color=DANGER, size=11), secondary_y=True)

    if fr is not None and np.isfinite(fr):
        i = int(np.argmin(c))
        fig.add_vline(x=fr, line=dict(color=ACCENT, width=1.6),
                      annotation_text=f"fr {fr:.3f} GHz",
                      annotation_position="top left",
                      annotation_font=dict(color=ACCENT, size=11))
        fig.add_trace(go.Scatter(
            x=[f[i]], y=[c[i]], mode="markers", name="resonance",
            marker=dict(color=ACCENT, size=11, line=dict(color="#1b1f27", width=1.6)),
            hovertemplate=f"fr = {fr:.4f} GHz<br>S11min = {c[i]:.2f} dB<extra></extra>",
        ), secondary_y=False)
        if bw and np.isfinite(bw) and bw > 0:
            fig.add_vrect(x0=fr - bw / 2, x1=fr + bw / 2,
                          fillcolor="rgba(255,152,56,0.07)", line_width=0, layer="below")

    fig.update_layout(title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01))
    fig.update_xaxes(title_text="Frequency (GHz)")
    fig.update_yaxes(title_text="S11 (dB)", secondary_y=False)
    fig.update_yaxes(title_text="VSWR", secondary_y=True, range=[1, 6])
    return _axes(_style(fig, height=440))


def compare_fig(entries):
    """Overlay several designs' S11 curves. entries: list of dicts with
    keys label, f, c, and optionally fr."""
    fig = go.Figure()
    for i, e in enumerate(entries):
        col = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(
            x=np.asarray(e["f"], dtype=float), y=np.asarray(e["c"], dtype=float),
            mode="lines", name=str(e["label"]),
            line=dict(color=col, width=2.2),
            hovertemplate="%{x:.3f} GHz<br>S11 = %{y:.2f} dB<extra>" + str(e["label"]) + "</extra>",
        ))
    fig.add_hline(y=-10, line=dict(color=DANGER, width=1.3, dash="dash"),
                  annotation_text="-10 dB", annotation_font=dict(color=DANGER, size=11))
    fig.update_layout(title=dict(text="Design Comparison — S11", font=dict(size=14, color=TEXT), x=0.01))
    fig.update_xaxes(title_text="Frequency (GHz)")
    fig.update_yaxes(title_text="S11 (dB)")
    return _axes(_style(fig, height=430))


def design_space_fig(db, shape_id, shape_label=""):
    """Resonant frequency across the (Lp, Wp) design space for one shape.

    Built straight from the CST `fr (GHz)` column -- no model calls, so it is instant.
    Shows a user where the design space puts their target frequency before they commit
    to a geometry, which is the question the inverse solver answers one point at a time.
    """
    sub = db[db["Antenna_ID"] == shape_id]
    if sub.empty or "fr (GHz)" not in sub.columns:
        return None
    piv = sub.pivot_table(index="Wp", columns="Lp", values="fr (GHz)", aggfunc="mean")
    if piv.empty:
        return None
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(dtype=float),
        x=[float(v) for v in piv.columns],
        y=[float(v) for v in piv.index],
        colorscale=[[0, "#123047"], [0.45, "#2f7fb8"], [0.7, "#ffb454"], [1, "#ff5470"]],
        colorbar=dict(title=dict(text="fr (GHz)", font=dict(color=AXIS, size=11)),
                      tickfont=dict(color=AXIS, size=10), thickness=13, len=0.85),
        hovertemplate="Lp = %{x:.2f} mm<br>Wp = %{y:.2f} mm<br>fr = %{z:.3f} GHz<extra></extra>",
        connectgaps=True,
    ))
    fig.update_layout(title=dict(
        text=f"Design Space — resonant frequency{(' · ' + shape_label) if shape_label else ''}",
        font=dict(size=14, color=TEXT), x=0.01))
    fig.update_xaxes(title_text="Patch length Lp (mm)")
    fig.update_yaxes(title_text="Patch width Wp (mm)")
    return _axes(_style(fig, height=430))


def error_scatter_fig(true_vals, pred_vals, xlabel="CST (truth)", ylabel="Predicted",
                      title="Prediction vs. ground truth", unit=""):
    """Parity plot for the validation panel. Points on the diagonal are perfect."""
    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    if t.size < 2:
        return None
    lo = float(min(t.min(), p.min()))
    hi = float(max(t.max(), p.max()))
    pad = (hi - lo) * 0.05 or 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
        line=dict(color="#6b7280", width=1.4, dash="dash"), name="perfect", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=t, y=p, mode="markers", name="held-out points",
        marker=dict(color=S11_COLOR, size=7, opacity=0.75,
                    line=dict(color="rgba(255,255,255,0.25)", width=0.6)),
        hovertemplate=f"truth %{{x:.3f}}{unit}<br>pred %{{y:.3f}}{unit}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01))
    fig.update_xaxes(title_text=xlabel, range=[lo - pad, hi + pad])
    fig.update_yaxes(title_text=ylabel, range=[lo - pad, hi + pad])
    return _axes(_style(fig, height=400))


# ---------------------------------------------------------------------------
# 3D structure view
# ---------------------------------------------------------------------------
SUBSTRATE_H = 1.6      # mm, typical FR-4 thickness
COPPER = "#cd7f32"
GROUND = "#b08040"
FR4 = "#f5deb3"


def _box_mesh(w, l, h, color, opacity=0.55, name=""):
    """Triangulated box centred on x, spanning y in [0, l], z in [-h, 0]."""
    x = [-w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2]
    y = [0, 0, l, l, 0, 0, l, l]
    z = [-h, -h, -h, -h, 0, 0, 0, 0]
    i = [0, 0, 4, 4, 1, 1, 2, 2, 3, 3, 4, 5]
    j = [1, 3, 5, 7, 2, 5, 6, 7, 0, 7, 6, 2]
    k = [2, 7, 6, 6, 6, 6, 5, 4, 4, 6, 2, 3]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity,
                     name=name, flatshading=True, hoverinfo="skip", showlegend=False)


def _flat_mesh(poly, z, color, name="", opacity=1.0):
    """Fan-triangulate a 2D polygon and place it flat at height z."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return None
    n = len(p)
    i = [0] * (n - 2)
    j = list(range(1, n - 1))
    k = list(range(2, n))
    return go.Mesh3d(x=p[:, 0], y=p[:, 1], z=[z] * n, i=i, j=j, k=k,
                     color=color, opacity=opacity, name=name, flatshading=True,
                     hoverinfo="skip", showlegend=False)


def patch_3d_fig(antenna_id, lp, wp, shape_label=""):
    """The physical CPW structure: an FR-4 slab with the copper pattern on its top
    face. CPW is a planar technology -- ground, feed and patch all sit in the same
    plane -- so the third dimension is the substrate thickness and the viewing angle,
    which is what makes the layering and the feed gap legible."""
    from antenna_viz import patch_polygons
    g = patch_polygons(antenna_id, lp, wp)
    sw, sl = g["substrate"]

    fig = go.Figure()
    fig.add_trace(_box_mesh(sw, sl, SUBSTRATE_H, FR4, opacity=0.45, name="FR-4 substrate"))
    for poly in g["grounds"]:
        m = _flat_mesh(poly, 0.0, GROUND, "CPW ground")
        if m: fig.add_trace(m)
    m = _flat_mesh(g["feed"], 0.0, COPPER, "feed line")
    if m: fig.add_trace(m)
    for poly in g["patch"]:
        m = _flat_mesh(poly, 0.0, COPPER, "patch")
        if m: fig.add_trace(m)

    # Lp / Wp dimension markers floating above the patch
    lf = 15.0
    fig.add_trace(go.Scatter3d(
        x=[-wp / 2 - 6, -wp / 2 - 6], y=[lf, lf + lp], z=[3, 3],
        mode="lines+text", line=dict(color=ACCENT, width=5),
        text=[None, f"Lp {lp:.2f} mm"], textposition="top center",
        textfont=dict(color=ACCENT, size=12), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[-wp / 2, wp / 2], y=[lf + lp + 5, lf + lp + 5], z=[3, 3],
        mode="lines+text", line=dict(color=ACCENT, width=5),
        text=[None, f"Wp {wp:.2f} mm"], textposition="top center",
        textfont=dict(color=ACCENT, size=12), hoverinfo="skip", showlegend=False))

    fig.update_layout(
        title=dict(text=f"CPW structure{(' — ' + shape_label) if shape_label else ''}",
                   font=dict(size=14, color=TEXT), x=0.01),
        scene=dict(
            xaxis=dict(title="x (mm)", color=AXIS, gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)",
                       showbackground=False, tickfont=dict(size=10, color=AXIS)),
            yaxis=dict(title="y (mm)", color=AXIS, gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)",
                       showbackground=False, tickfont=dict(size=10, color=AXIS)),
            zaxis=dict(title="z (mm)", color=AXIS, gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)",
                       showbackground=False, tickfont=dict(size=10, color=AXIS)),
            aspectmode="manual", aspectratio=dict(x=1.1, y=1.3, z=0.45),
            camera=dict(eye=dict(x=1.5, y=-1.7, z=0.85)),
        ),
    )
    return _style(fig, height=520)
