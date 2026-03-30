# Song2Vec: Music Recommendation with Word Embeddings

A music recommendation system built with Word2Vec (Skip-gram with Negative Sampling), trained on the 30Music playlist dataset. Songs in playlists are treated as words in sentences — songs that frequently co-occur in playlists learn similar embeddings, enabling semantic similarity-based recommendations.

---

## Overview

This project implements and evaluates a Song2Vec system for next-song prediction. Given a playlist of N songs, the model predicts the N-th song using the preceding songs as context.

Two evaluation approaches are compared:
- **Context Averaging**: Average embedding of all N−1 context songs used as query
- **Single Query**: Only the (N−1)-th song used as query (Paper approach)

---

## Dataset

The [30Music Dataset](https://remaplab.deib.polimi.it/resources/) — a collection of user-created playlists retrieved from Last.fm.

| Statistic | Value |
|-----------|-------|
| Total playlists | 47,130 |
| Train / Test split | 80% / 20% |
| Total track occurrences | 1,601,748 |
| Unique tracks | 443,184 |
| Unique artists | 60,282 |

Download `playlist.idomaar` and `tracks.idomaar` from the dataset source and set `BASE_PATH` in the notebook accordingly.

---

## Results

| Model | Approach | HR@5 | HR@10 | HR@20 | HR@50 |
|-------|----------|------|-------|-------|-------|
| Baseline | Context Avg. | 0.0166 | 0.0249 | 0.0350 | 0.0503 |
| Baseline | Single Query | 0.0214 | 0.0331 | 0.0502 | 0.0706 |
| **Final** | **Context Avg.** | **0.0281** | **0.0491** | **0.0747** | **0.1192** |
| **Final** | **Single Query** | **0.0559** | **0.0786** | **0.1107** | **0.1515** |

Final model achieves **2.4× improvement** over baseline at HR@10 (Single Query).

---

## Final Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Skip-gram (sg=1) |
| Vector size | 100 |
| Window size | 40 |
| Min count | 5 |
| Negative samples | 20 |
| ns_exponent | 0.75 |
| Epochs | 20 |

---

## Key Findings

- **Window size**: Larger windows (40) significantly improve performance. Unlike NLP, broader playlist context is more informative for recommendation.
- **ns_exponent**: Contrary to the paper's recommendation, positive values (0.75) outperform negative values on this dataset due to the flatter popularity distribution in playlist data vs. listening sessions.
- **Epochs**: Performance peaks at 20 epochs; beyond that, marginal gains with risk of overfitting.
- **Single Query > Context Averaging**: Using the most recent song as a single query preserves specificity; averaging dilutes the embedding.

---

## Qualitative Results

**Most similar songs** (seed: `led zeppelin__stairway to heaven`):
```
1. led zeppelin__whole lotta love     (0.7674)
2. led zeppelin__black dog            (0.7663)
3. led zeppelin__immigrant song       (0.7502)
7. queen__bohemian rhapsody           (0.6849)  ← cross-artist
8. pink floyd__comfortably numb       (0.6755)  ← cross-artist
```

**Song algebra** (analogous to king − man + woman = queen):
```
radiohead__creep − radiohead__karma police + the beatles__let it be
= the beatles__yesterday (0.7576)
```

---

## Setup

```bash
pip install gensim scikit-learn numpy pandas matplotlib
```

Open `Song2Vec.ipynb` in Jupyter or Google Colab. Set `BASE_PATH` to the directory containing `playlist.idomaar` and `tracks.idomaar`.

---

## Structure

```
Song2Vec.ipynb          # Main notebook
README.md               # This file
```

Saved model files (`baseline_model.model`, `final_model.model`) and processed data pickles are not included due to size.

---

## References

- Caselles-Dupré, H., Lesaint, F., & Royo-Letelier, J. (2018). *Word2Vec applied to Recommendation: Hyperparameters Matter*. RecSys '18.
- Turrin, R., et al. (2015). *30Music Listening and Playlists Dataset*. RecSys 2015 Workshop.
