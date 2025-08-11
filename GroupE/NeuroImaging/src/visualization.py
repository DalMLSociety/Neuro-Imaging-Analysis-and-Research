# visualization.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io
import itertools
from config import opt

def plot_3d_graph(conn:np.ndarray, comm_vec=None, thresh:float=.3):
    """
    Create a 3‑D network visualization with Plotly.
    Edges below |thresh| are hidden for clarity.
    """
    if not opt.get('plotly', False):
        st.warning("Plotly not installed ➜ 3D visualization is disabled"); return
    import plotly.graph_objects as go
    from utils import random_sphere_coords
    N = conn.shape[0]
    coords = random_sphere_coords(N)
    edge_x, edge_y, edge_z = [], [], []
    for i, j in itertools.combinations(range(N), 2):
        w = conn[i,j]
        if abs(w) < thresh: continue
        edge_x += [coords[i,0], coords[j,0], None]
        edge_y += [coords[i,1], coords[j,1], None]
        edge_z += [coords[i,2], coords[j,2], None]
    node_x, node_y, node_z = coords[:,0], coords[:,1], coords[:,2]
    node_color = comm_vec if comm_vec is not None else np.ones(N)
    fig = go.Figure(data=[
        go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
            mode='lines', line=dict(width=2), hoverinfo='none'),
        go.Scatter3d(x=node_x, y=node_y, z=node_z,
            mode='markers', marker=dict(size=5,
            color=node_color, colorscale='Viridis', showscale=True))
    ])
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

def make_dfc_movie(dfc_stack: np.ndarray, out_path: str = "dfc_anim.mp4", fps: int = 2):
    if not opt.get('moviepy', False):
        st.warning("moviepy not installed ➜ dFC animation disabled")
        return None
    import moviepy.editor as mpy
    clips = []
    vmax = np.max(np.abs(dfc_stack))
    for m in dfc_stack:
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(m, vmin=-vmax, vmax=vmax, cmap='coolwarm', cbar=False, ax=ax)
        ax.axis('off')
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        plt.close('all')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_np = np.array(img)
        clips.append(mpy.ImageClip(img_np).set_duration(1/fps))
    concat = mpy.concatenate_videoclips(clips, method="compose")
    concat.write_videofile(out_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
    with open(out_path, "rb") as f:
        return f.read()
