import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load model
clusters = joblib.load("kmeans_custom.pkl")
k = len(clusters)

# Dataset
X, _ = make_blobs(n_samples=200, n_features=2, centers=4, random_state=23)

# Distance
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# -------- INITIAL GRAPH --------
def plot_initial():
    fig, ax = plt.subplots(figsize=(4,4))  # smaller size

    ax.scatter(X[:, 0], X[:, 1], alpha=0.3)

    for i in range(k):
        center = clusters[i]['center']
        ax.scatter(center[0], center[1], marker='X', s=150)

    ax.set_title("Clusters Overview")
    ax.grid(True)

    return fig

# -------- PREDICT --------
def predict_and_plot(x, y):
    point = np.array([x, y])

    distances = [distance(point, clusters[i]['center']) for i in range(k)]
    cluster_id = int(np.argmin(distances))

    fig, ax = plt.subplots(figsize=(4,4))  # smaller

    ax.scatter(X[:, 0], X[:, 1], alpha=0.3)

    for i in range(k):
        center = clusters[i]['center']
        ax.scatter(center[0], center[1], marker='X', s=150)

    ax.scatter(point[0], point[1], s=100)

    ax.set_title(f"Cluster {cluster_id}")
    ax.grid(True)

    return f"Cluster: {cluster_id}", fig


# -------- UI --------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("## ✨ Customer Segmentation using K-Means")
    gr.Markdown("Enter values to see cluster assignment visually")

    with gr.Row():

        # LEFT PANEL (INPUT)
        with gr.Column(scale=1):
            gr.Markdown("### Input")

            x_input = gr.Number(label="x")
            y_input = gr.Number(label="y")

            submit_btn = gr.Button("Predict", variant="primary")

            output_text = gr.Textbox(label="Result")

        # RIGHT PANEL (GRAPHS)
        with gr.Column(scale=2):
            gr.Markdown("### Visualization")

            with gr.Row():
                initial_plot = gr.Plot(label="Before")
                output_plot = gr.Plot(label="After")

    # Load initial graph
    demo.load(fn=plot_initial, outputs=initial_plot)

    # On click
    submit_btn.click(
        fn=predict_and_plot,
        inputs=[x_input, y_input],
        outputs=[output_text, output_plot]
    )

demo.launch()