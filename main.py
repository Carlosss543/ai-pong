import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo
import os


checkpoint_path = "./training_checkpoints/checkpoint.pkl"
resume_training = False

num_episodes = 1000
batch_size = 1
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_img(img):
    """ process the input tensor of shape (210, 160, 3) to turn it into a tensor of shape (1, 6400) """
    img = torch.from_numpy(img) if not isinstance(img, torch.Tensor) else img
    img = img[34:194] # crop the upper and lower part
    img = img[::2, ::2, 0] # downsample by factor of 2 and take only the first channel
    img[(img == 109) | (img == 144)] = 0 # erase background
    img[img != 0] = 1 # everything else (paddles, ball) just set to 1
    return img.reshape(1, -1).to(torch.float32)


def discount_rewards(policy_rewards, gamma=0.99):
    R = 0
    rewards = []
    for r in reversed(policy_rewards):
        if r != 0: # non-zero rewards indicate the end of a point
            R = 0
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return rewards


class Policy(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(80*80, 256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def main():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", repeat_action_probability=0.0, render_mode="rgb_array")
    os.makedirs("videos", exist_ok=True)
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: ep % 200 == 0, name_prefix="pong")

    policy = Policy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

    start_episode = 1
    running_reward = -21.0

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"] + 1
        running_reward = checkpoint["running_reward"]

    log_probs_list, rewards_list = [], []

    for episode in range(start_episode, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_rewards = 0

        while not done:
            x = process_img(state).to(device)

            probs = policy(x)

            distribution = Categorical(probs)
            action = distribution.sample()

            state, reward, terminated, truncated, _ = env.step(action.item() + 2)
            done = terminated or truncated

            log_probs_list.append(distribution.log_prob(action))
            rewards_list.append(reward)

            episode_rewards += reward

        running_reward = running_reward * 0.99 + episode_rewards * 0.01
        print(f"Episode:{episode} \t Average reward: {running_reward:.2f} \t Episode reward: {episode_rewards:.2f}")

        if episode % batch_size == 0:
            discounted_rewards = discount_rewards(rewards_list)
            loss = -(torch.cat(log_probs_list) * discounted_rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_probs_list, rewards_list = [], []

        if episode % 100 == 0 or episode == num_episodes:
            print("ep %d: model saving..." % episode)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "episode": episode,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "running_reward": running_reward,
            }, checkpoint_path)

    env.close()


if __name__ == "__main__":
    main()
