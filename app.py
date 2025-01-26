import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

# Streamlit layout setting
st.set_page_config(layout="wide")  # Use the full width of the screen

# Streamlit setup
st.title("Interactive Linear Regression with Plotly")
st.write("This app demonstrates linear regression training with Plotly visualizations, including the 3D error surface and dynamic updates.")

# Sidebar inputs
st.sidebar.header("Data & Training Options")
data_size = st.sidebar.slider("Number of data points", 20, 200, 100)
noise_level = st.sidebar.slider("Noise level", 0.0, 5.0, 1.0)
learning_rate = st.sidebar.slider(
    "Learning rate", 0.001, 0.1, 0.01, step=0.001)
epochs = st.sidebar.slider("Number of epochs", 10, 200, 50)

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, data_size)
y = 3 * X + 5 + np.random.normal(0, noise_level, data_size)

# Normalize features
X = (X - np.mean(X)) / np.std(X)
X = X.reshape(-1, 1)
X = np.c_[np.ones(X.shape[0]), X]  # Add bias term (X0 = 1)

# Initialize parameters
theta = np.random.randn(2)
loss_history = []
theta_history = []

# Gradient descent function


def gradient_descent_step(X, y, theta, learning_rate):
    m = len(y)
    y_pred = np.dot(X, theta)
    error = y_pred - y
    gradient = (1 / m) * np.dot(X.T, error)
    theta -= learning_rate * gradient
    loss = mean_squared_error(y, y_pred)
    return theta, loss, y_pred


# Precompute error surface
W_range = np.linspace(-1, 5, 100)
B_range = np.linspace(-1, 10, 100)
W, B = np.meshgrid(W_range, B_range)
loss_surface = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        w = W[i, j]
        b = B[i, j]
        y_pred = w * X[:, 1] + b
        loss_surface[i, j] = mean_squared_error(y, y_pred)

# "Start" button
if st.button("Start Training"):
    st.subheader("Training Visualization")

    # Create columns to hold each plot
    col1, col2, col3 = st.columns(3)

    # Create empty placeholders for the plots
    plot_area_line = col1.empty()
    plot_area_loss = col2.empty()
    plot_area_surface = col3.empty()

    progress_bar = st.progress(0)

    for epoch in range(epochs):
        # Perform one step of gradient descent
        theta, loss, y_pred = gradient_descent_step(X, y, theta, learning_rate)
        loss_history.append(loss)
        theta_history.append(theta.copy())

        # Plot 1: Regression Line Update
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=X[:, 1], y=y, mode="markers", name="Data"))
        fig_line.add_trace(go.Scatter(
            x=X[:, 1], y=y_pred, mode="lines", name="Regression Line", line=dict(color="red")))
        fig_line.update_layout(
            title=f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}",
            xaxis_title="Feature (X)",
            yaxis_title="Target (y)",
            # Increased top margin for the title
            margin=dict(t=40, b=0, l=40, r=40)
        )
        plot_area_line.plotly_chart(fig_line, use_container_width=True)

        # Plot 2: Loss Reduction
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(range(1, epoch + 2)),
                           y=loss_history, mode="lines+markers", name="Loss"))
        fig_loss.update_layout(
            title="Error Reduction Over Epochs",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            # Increased top margin for the title
            margin=dict(t=40, b=0, l=40, r=40)
        )
        plot_area_loss.plotly_chart(fig_loss, use_container_width=True)

        # Plot 3: 3D Error Surface with Gradient Descent Path
        fig_surface = go.Figure()

        # Plot full 3D parabolic surface
        fig_surface.add_trace(
            go.Surface(z=loss_surface, x=W, y=B, colorscale="Viridis",
                       opacity=0.8, name="Error Surface")
        )

        # Plot gradient descent path
        theta_array = np.array(theta_history)
        fig_surface.add_trace(
            go.Scatter3d(
                x=theta_array[:, 1],
                y=theta_array[:, 0],
                z=loss_history,
                mode="markers+lines",
                marker=dict(color="red", size=5),
                name="Gradient Descent Path",
            )
        )

        fig_surface.update_layout(
            title="3D Error Surface",
            scene=dict(
                xaxis_title="Weight (W)",
                yaxis_title="Bias (B)",
                zaxis_title="Loss",
            ),
            # Increased top margin for the title
            margin=dict(t=40, b=0, l=40, r=40)
        )
        plot_area_surface.plotly_chart(fig_surface, use_container_width=True)

        # Update progress bar
        progress_bar.progress((epoch + 1) / epochs)

    # Display final parameters and loss
    st.write("### Training Complete!")
    st.write(f"**Final Parameters:** θ0 = {theta[0]:.4f}, θ1 = {theta[1]:.4f}")
    st.write(f"**Final Loss:** {loss:.4f}")
