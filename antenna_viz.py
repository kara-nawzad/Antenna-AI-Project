import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_antenna(antenna_id, lp, wp):
    # Set high resolution
    fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
    
    # --- TECHNICAL DESIGN PALETTE ---
    substrate_color = '#F5DEB3' # FR-4 Wheat
    copper_color = '#CD7F32'    # Metallic Copper
    shadow_color = '#A9A9A9'    # Simple Gray Shadow
    
    lf, wf, g, ws, ls = 15.0, 3.0, 0.5, 50.0, 60.0
    g_ymax = 12.0 # Grounds end at 12mm (lf-3)

    # 1. DRAW SUBSTRATE
    ax.add_patch(patches.Rectangle((-ws/2, 0), ws, ls, color=substrate_color, ec='#8B7355', lw=1, zorder=1))
    
    # 2. HELPER FUNCTION (Manual Shadow + Metal)
    def add_metal(patch_obj):
        # Create a manual shadow by shifting the shape
        if isinstance(patch_obj, patches.Rectangle):
            shadow = patches.Rectangle((patch_obj.get_x()+0.3, patch_obj.get_y()-0.3), 
                                      patch_obj.get_width(), patch_obj.get_height(), color=shadow_color, zorder=2)
        elif isinstance(patch_obj, patches.Polygon):
            shadow = patches.Polygon(patch_obj.get_xy() + [0.3, -0.3], color=shadow_color, zorder=2)
        elif isinstance(patch_obj, patches.Ellipse):
            shadow = patches.Ellipse((patch_obj.center[0]+0.3, patch_obj.center[1]-0.3), 
                                    patch_obj.width, patch_obj.height, color=shadow_color, zorder=2)
        
        ax.add_patch(shadow)
        patch_obj.set_facecolor(copper_color)
        patch_obj.set_edgecolor('#5D2906')
        patch_obj.set_linewidth(1)
        patch_obj.set_zorder(3)
        ax.add_patch(patch_obj)

    # 3. DRAW GROUNDS AND FEED
    gw = (ws/2) - (wf/2) - g
    add_metal(patches.Rectangle((-ws/2, 0), gw, g_ymax)) # Left
    add_metal(patches.Rectangle((wf/2 + g, 0), gw, g_ymax)) # Right
    add_metal(patches.Rectangle((-wf/2, 0), wf, lf)) # Feed

    # 4. THE 12 ANTENNA SHAPES (EXACT CST REPLICAS)
    if antenna_id == 1: # Rectangle
        add_metal(patches.Rectangle((-wp/2, lf), wp, lp))
        
    elif antenna_id == 2: # Stepped
        add_metal(patches.Rectangle((-7.5, lf), 15, lp/2))
        add_metal(patches.Rectangle((-wp/2, lf + lp/2), wp, lp/2))
        
    elif antenna_id == 3: # T-Shape
        add_metal(patches.Rectangle((-1.5, lf), 3, lp/2))
        add_metal(patches.Rectangle((-wp/2, lf + lp/2), wp, lp/2))
        
    elif antenna_id == 4: # Ellipse
        add_metal(patches.Ellipse((0, lf + lp/2), wp, lp))
        
    elif antenna_id == 5: # Semi-Ellipse
        theta = np.linspace(0, np.pi, 50)
        x = (wp/2) * np.cos(theta)
        y = lf + lp * np.sin(theta)
        pts = np.column_stack([x, y])
        pts = np.vstack([pts, [wp/2, lf], [-wp/2, lf]])
        add_metal(patches.Polygon(pts))

    elif antenna_id == 6: # Pie-Sector (Curved Fan)
        # 1. Create the arc for the top of the "fan"
        # theta goes from 0 (Right) to pi (Left)
        theta = np.linspace(0, np.pi, 50)
        
        # Width controlled by wp
        x_arc = (wp/2) * np.cos(theta)
        
        # Height controlled by lp (Top arc starts at 60% of lp height)
        y_arc = (lf + lp * 0.6) + (lp * 0.4) * np.sin(theta)
        pts_arc = np.column_stack([x_arc, y_arc])
        
        # 2. COMBINE POINTS IN CORRECT CIRCULAR ORDER
        # Order: Bottom-Right -> Arc (Right to Left) -> Bottom-Left
        # This prevents the lines from crossing (the hourglass error)
        pts = np.vstack([
            [1.5, lf],      # Bottom-Right: Connects to feed line top-right corner
            pts_arc,        # The entire curved top (50 points)
            [-1.5, lf]      # Bottom-Left: Connects to feed line top-left corner
        ])
        
        add_metal(patches.Polygon(pts))

    elif antenna_id == 7: # Triangle
        add_metal(patches.Polygon([[-wp/2, lf], [wp/2, lf], [0, lf+lp]]))

    elif antenna_id == 8: # Trapezoid
        add_metal(patches.Polygon([[-1.5, lf], [1.5, lf], [wp/2, lf+lp], [-wp/2, lf+lp]]))

    elif antenna_id == 9: # Diamond
        add_metal(patches.Polygon([[-1.5, lf], [1.5, lf], [wp/2, lf+lp/2], [0, lf+lp], [-wp/2, lf+lp/2]]))

    elif antenna_id == 10: # Hexagon
        y1, y2 = lf + lp/3, lf + 2*lp/3
        pts = [[-1.5, lf], [1.5, lf], [wp/2, y1], [wp/2, y2], [1.5, lf+lp], [-1.5, lf+lp], [-wp/2, y2], [-wp/2, y1]]
        add_metal(patches.Polygon(pts))

    elif antenna_id == 11: # Pentagon
        add_metal(patches.Polygon([[-wp/2, lf], [wp/2, lf], [wp/2, lf+lp/2], [0, lf+lp], [-wp/2, lf+lp/2]]))

    elif antenna_id == 12: # Cross
        add_metal(patches.Rectangle((-1.5, lf), 3, lp))
        add_metal(patches.Rectangle((-wp/2, lf + lp/2 - 1.5), wp, 3))

    # 5. PROFESSIONAL DIMENSIONS (Technical Boxes)
    def add_label(start, end, text, is_vert=True):
        col = '#2C3E50'
        if is_vert:
            ax.annotate('', xy=(start[0], start[1]), xytext=(end[0], end[1]), arrowprops=dict(arrowstyle='<->', color=col))
            ax.text(start[0]+2, (start[1]+end[1])/2, text, rotation=90, va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1), fontsize=8)
        else:
            ax.annotate('', xy=(start[0], start[1]), xytext=(end[0], end[1]), arrowprops=dict(arrowstyle='<->', color='#C0392B'))
            ax.text((start[0]+end[0])/2, start[1]-4, text, ha='center', fontweight='bold', color='#C0392B',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#C0392B', lw=1), fontsize=8)

    add_label((wp/2 + 8, lf), (wp/2 + 8, lf + lp), f"Lp: {lp}mm")
    add_label((-wp/2, lf - 8), (wp/2, lf - 8), f"Wp: {wp}mm", is_vert=False)

    # Final settings
    ax.set_xlim(-30, 30); ax.set_ylim(-5, 65)
    ax.set_aspect('equal'); ax.axis('off')
    plt.title(f"AI PREDICTED DESIGN: ANTENNA {antenna_id}", fontsize=12, fontweight='bold', pad=20)
    return fig

if __name__ == "__main__":
    # Test the Stepped Rectangle (ID 2)
    draw_antenna(12, 15, 30)
    plt.show()

# ---------------------------------------------------------------------------
# Geometry as data
#
# The 12 patch outlines expressed as polygons, so they can be rendered by
# something other than matplotlib (see antenna_charts.patch_3d_fig). The
# constants and vertex definitions are identical to draw_antenna() above.
# ---------------------------------------------------------------------------
LF, WF, GAP, WS, LS = 15.0, 3.0, 0.5, 50.0, 60.0
G_YMAX = 12.0


def _rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def patch_polygons(antenna_id, lp, wp):
    """Return the metal outlines for a given design.

    dict with keys:
      'patch'     list of (N,2) arrays -- the radiating element
      'grounds'   list of (N,2) arrays -- the two CPW ground planes
      'feed'      (N,2) array          -- the feed line
      'substrate' (width, length)      -- the FR-4 slab footprint
    """
    gw = (WS / 2) - (WF / 2) - GAP
    grounds = [_rect(-WS / 2, 0, gw, G_YMAX), _rect(WF / 2 + GAP, 0, gw, G_YMAX)]
    feed = _rect(-WF / 2, 0, WF, LF)

    lf, wp, lp = float(LF), float(wp), float(lp)
    aid = int(antenna_id)
    patch = []

    if aid == 1:                                     # Rectangle
        patch = [_rect(-wp / 2, lf, wp, lp)]
    elif aid == 2:                                   # Stepped
        patch = [_rect(-7.5, lf, 15, lp / 2), _rect(-wp / 2, lf + lp / 2, wp, lp / 2)]
    elif aid == 3:                                   # T-Shape
        patch = [_rect(-1.5, lf, 3, lp / 2), _rect(-wp / 2, lf + lp / 2, wp, lp / 2)]
    elif aid == 4:                                   # Ellipse
        t = np.linspace(0, 2 * np.pi, 90)
        patch = [np.column_stack([(wp / 2) * np.cos(t) + 0.0,
                                  lf + lp / 2 + (lp / 2) * np.sin(t)])]
    elif aid == 5:                                   # Semi-Ellipse
        t = np.linspace(0, np.pi, 50)
        pts = np.column_stack([(wp / 2) * np.cos(t), lf + lp * np.sin(t)])
        patch = [np.vstack([pts, [[wp / 2, lf], [-wp / 2, lf]]])]
    elif aid == 6:                                   # Pie-Sector
        t = np.linspace(0, np.pi, 50)
        arc = np.column_stack([(wp / 2) * np.cos(t),
                               (lf + lp * 0.6) + (lp * 0.4) * np.sin(t)])
        patch = [np.vstack([[1.5, lf], arc, [-1.5, lf]])]
    elif aid == 7:                                   # Triangle
        patch = [np.array([[-wp / 2, lf], [wp / 2, lf], [0, lf + lp]], dtype=float)]
    elif aid == 8:                                   # Trapezoid
        patch = [np.array([[-1.5, lf], [1.5, lf], [wp / 2, lf + lp], [-wp / 2, lf + lp]], dtype=float)]
    elif aid == 9:                                   # Diamond
        patch = [np.array([[-1.5, lf], [1.5, lf], [wp / 2, lf + lp / 2],
                           [0, lf + lp], [-wp / 2, lf + lp / 2]], dtype=float)]
    elif aid == 10:                                  # Hexagon
        y1, y2 = lf + lp / 3, lf + 2 * lp / 3
        patch = [np.array([[-1.5, lf], [1.5, lf], [wp / 2, y1], [wp / 2, y2],
                           [1.5, lf + lp], [-1.5, lf + lp], [-wp / 2, y2], [-wp / 2, y1]], dtype=float)]
    elif aid == 11:                                  # Pentagon
        patch = [np.array([[-wp / 2, lf], [wp / 2, lf], [wp / 2, lf + lp / 2],
                           [0, lf + lp], [-wp / 2, lf + lp / 2]], dtype=float)]
    elif aid == 12:                                  # Cross
        patch = [_rect(-1.5, lf, 3, lp), _rect(-wp / 2, lf + lp / 2 - 1.5, wp, 3)]

    return {"patch": patch, "grounds": grounds, "feed": feed,
            "substrate": (float(WS), float(LS))}
