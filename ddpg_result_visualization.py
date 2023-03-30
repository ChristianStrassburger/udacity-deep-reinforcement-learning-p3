import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, moving_avg_scores):
    """Plots a score list.
        
        Params
        ======
        scores (array): a score list
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.plot(np.arange(1, len(moving_avg_scores)+1), moving_avg_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def format_episode_score(i_episode, episode_score, moving_avg_score):
    """Returns a formatted episode result.
        
        Params
        ======
        i_episode (int): episode index
        episode_score (float): episode score
        moving_avg_score (float): moving average score
    """
    return f"Episode: {i_episode}\tEpisode score: {episode_score:.4f}\tMoving average score: {moving_avg_score:.4f}"