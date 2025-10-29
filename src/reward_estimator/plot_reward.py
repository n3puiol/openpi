import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_values(ep_start, ep_end, skip, rewards, episode):
    x_steps = np.array(list(range(ep_start, ep_end, skip)))
    y_vals = np.asarray([float(r) for r in rewards])

    window_size = 20
    reward_series = pd.Series(y_vals)
    smoothed_rewards = reward_series.rolling(window=window_size).mean()

    x_steps = x_steps - x_steps[0]
    normalized_rewards = (smoothed_rewards - np.min(smoothed_rewards)) / (np.max(smoothed_rewards) - np.min(smoothed_rewards))

    # plt.plot(x_steps, y_vals, marker="o", label="Regularized Reward")
    plt.plot(
        x_steps,
        # smoothed_rewards,
        normalized_rewards,
        linestyle="-",
        color="orange",
        label="Estimated Reward",
    )

    # # Fit and plot a linear regression line
    # if len(x_steps) >= 2:
    #     coeffs = np.polyfit(
    #         x_steps, y_vals, 1
    #     )  # coeffs[0] = slope, coeffs[1] = intercept
    #     y_fit = np.polyval(coeffs, x_steps)
    #     plt.plot(
    #         x_steps,
    #         y_fit,
    #         linestyle="--",
    #         color="red",
    #         label=f"Linear fit (slope={coeffs[0]:.4f})",
    #     )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Estimated Reward over Episode")
    plt.grid()
    plt.legend()
    plt.savefig(
        f"regularized_reward_plots/smoothed_regularized_reward_plot_{episode}.png"
    )


if __name__ == "__main__":
    ep_start = 1925
    ep_end = 2195
    skip = 5
    rewards = [
        -3.125,
        -2.35938,
        -1.90625,
        -1.75,
        0.0429688,
        -1.70312,
        -0.148438,
        0.632812,
        0.699219,
        0.726562,
        0.648438,
        -4.09375,
        -5.34375,
        -1.54688,
        -0.585938,
        -1.875,
        -0.734375,
        -0.3125,
        -1.8125,
        -1.20312,
        -1.14062,
        -0.09375,
        -0.359375,
        0.792969,
        0.730469,
        0.734375,
        0.210938,
        0.597656,
        0.164062,
        0.101562,
        -4.5,
        -4.40625,
        -4.78125,
        -3.59375,
        -1.76562,
        -3.5625,
        -1.89062,
        -3.46875,
        -0.375,
        -0.414062,
        -0.320312,
        -0.445312,
        0.164062,
        -1.29688,
        -2.67188,
        -2.625,
        -1.01562,
        0.730469,
        0.394531,
        0.28125,
        -0.96875,
        -4.53125,
        -8.3125,
        -1.20312,
    ]
    episode = 7
    plot_values(ep_start, ep_end, skip, rewards, episode)
