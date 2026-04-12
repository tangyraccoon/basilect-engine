# Bibliography — Landscape of Music Artist Similarity Research

Sources from the deep research document + supplementary searches. Grouped by my notes first, then sorted roughly by how much weight they carry as precedent for the project.

---

## Sources I flagged in my notes

### HDSR influence graph (closest precedent)
- **Badillo-Goicoechea, E. (2025).** "Modeling Artist Influence for Music Selection and Recommendation: A Purely Network-Based Approach." *Harvard Data Science Review*, Issue 7.4, Fall 2025. https://hdsr.mitpress.mit.edu/pub/t4txmd81/release/2
  - 22,831 artists, 159,389 edges from critic namedrops (Pitchfork, The Quietus, NPR). BFS/Dijkstra/max-flow graph traversal for recs. Post-hoc sonic similarity via Spotify audio features. First system explicitly targeting "bridge artist discovery."

### Cosine.club (sonic embedding precedent)
- **Cosine.club (2024).** Music Similarity Search Engine. https://cosine.club
  - Uses Essentia/UPF's Discogs-EfficientNet model. 1.15M electronic music tracks. Deliberately breaks popularity bias. Electronic music only. Ethos: uncovering hidden gems, not hidden connections.

### Heterogeneous GNNs for artist similarity
- **Da Silva, J.M. et al. (2024).** "Artist Similarity Based on Heterogeneous Graph Neural Networks." *IEEE/ACM Transactions on Audio, Speech and Language Processing*, August 2024. https://dl.acm.org/doi/10.1109/TASLP.2024.3437170
  - First multimodal heterogeneous graph integrating audio, lyrics, and artist relations for artist similarity via link prediction.

- **GATSY — Ferraro et al. (2024–2025).** "GATSY: Graph Attention Network for Music Artist Similarity." Sapienza/ISTI-CNR. https://arxiv.org/html/2311.00635v2
  - Introduced "fictitious artists" as bridge nodes. **Key finding: genre-based supervision does not improve artist similarity.** Uses OLGA dataset with MusicBrainz genre labels (2,842 genres).

- **MUSYNERGY (2025).** "A framework for music collaboration discovery based on neural networks and graph analysis." *International Journal of Entertainment Technology and Management*. https://www.sciencedirect.com/science/article/pii/S1875952125001132
  - Heterogeneous Knowledge Graph from MusicBrainz. Predicts potential creative partnerships, not just similarity. Link prediction task.

### MusicLynx (metadata-based non-obvious connections)
- **Allik, A. et al. (2018).** "MusicLynx: Exploring Music Through Artist Similarity Graphs." *WWW '18 Companion*. https://dl.acm.org/doi/fullHtml/10.1145/3184558.3186970
  - Finds related artists via any shared metadata: ethnic background, nationality, record label, "shared fate or affliction." Closest to my approach in spirit. Uses DBpedia/Wikidata.

### MoodPlay (2D mood space rec system)
- **Andjelkovic, I., Parra, D., & O'Donovan, J. (2016/2019).** "MoodPlay: Interactive Mood-based Music Discovery and Recommendation." *UMAP 2016*; full paper in *Int. J. Human-Computer Studies*, Vol. 121, pp. 142–159, 2019. https://dl.acm.org/doi/10.1145/2930238.2930280
  - 2D mood space from Last.fm tags (valence: negative↔positive, arousal: calm↔excited). GEMS affect model. N=240 user study. Hybrid content + mood filtering.

- **Allik, A., Thalmann, F., Metzig, C., & Sandler, M. (2019).** "moodplay.github.io: an online collaborative music player." *WAC 2019*, Queen Mary University of London. https://www.ntnu.edu/documents/1282113268/1290817988/WAC2019-CameraReadySubmission-23.pdf
  - The web implementation. Auto-DJ module, semantic mood space from Last.fm tags via Self-Organizing Maps.

---

## High-weight precedent (directly relevant to the project's direction)

### Foundational artist similarity & ground truth
- **Ellis, D., Whitman, B., Berenzweig, A., & Lawrence, S. (2002).** "The Quest for Ground Truth in Musical Artist Similarity." *Proc. ISMIR 2002*. http://labrosa.ee.columbia.edu/projects/musicsim/
  - Compared acoustic, collaborative, playlist, and editorial similarity on 400 artists. The paper that established the multi-signal comparison framework — but stopped short of measuring orthogonality.

- **Whitman, B. & Lawrence, S. (2002).** "Inferring Descriptions and Similarity for Music from Community Metadata." MIT Media Lab. 
  - Built TF·IDF artist profiles from web text. The original NLP-for-artist-similarity paper.

### Text/NLP for artist similarity
- **Oramas, S. et al. (2015).** "A Semantic-based Approach for Artist Similarity." *ISMIR 2015*. https://archives.ismir.net/ismir2015/paper/000305.pdf
  - Entity linking on Last.fm biographies via Babelfy → BabelNet/DBpedia. Semantic graphs outperformed surface text and word co-occurrence.

### Surveys (landscape context)
- **Schedl, M., Knees, P., McFee, B., Bogdanov, D., & Kaminskas, M. (2013).** "A Survey of Music Similarity and Recommendation from Music Context Data." *ACM Transactions on Multimedia Computing, Communications, and Applications*, Vol. 10, No. 1. https://dl.acm.org/doi/pdf/10.1145/2542205.2542206
  - The definitive survey of context-based approaches: text-retrieval, co-occurrence, user-rating. Predicted multi-faceted similarity would become standard (still hasn't happened 12 years later).

- **Deldjoo, Y. et al. (2024).** Systematic review on MRS (referenced in landscape paper for "onion model" of five content layers). 
  - Signal, embedded metadata, expert-generated content, user-generated content, derivative content. No system fuses all five.

- **Epure, E.V., Deldjoo, Y., Sguerra, B., Schedl, M., & Moussallam, B. (2025).** "Music Recommendation with Large Language Models." *arXiv*, November 2025. https://arxiv.org/pdf/2511.16478
  - Definitive survey of LLM-for-music-rec. Two paradigms: LLM as ranker vs. LLM as embedding model.

### Influence/sampling networks
- **Bryan, N.J. & Wang, G. (2011).** "Musical Influence Network Analysis and Rank of Sample-Based Music." *ISMIR 2011*. Stanford CCRMA. https://ccrma.stanford.edu/~njb/research/influence.pdf
  - WhoSampled network analysis. Funk/soul/disco dominance as source material for hip-hop. Network analysis only, no recommender built.

- **Figueiredo, F. & Andrade, N. (2019).** "Quantifying Disruptive Influence in the AllMusic Guide." *ISMIR 2019*. https://archives.ismir.net/ismir2019/paper/000102.pdf
  - AllMusic influence graph analysis. "Disruptive" vs. "consolidating" artists.

- **Park et al. (2024).** "Surprising Patterns in Musical Influence Networks." *arXiv*. https://arxiv.org/html/2410.15996v1
  - Bayesian Surprise applied to temporal evolution of influence networks.

### Temporal/trajectory approaches
- **Collins, T. (2025).** "Recording artist career comparison through audio content analysis." *Royal Society Open Science*. https://royalsocietypublishing.org/doi/10.1098/rsos.241647
  - R.E.M., Radiohead, Coldplay career trajectories via timbral-rhythmic and harmonic features. The sole direct precedent for trajectory convergence.

- **Mauch, M. et al. (2015).** "The Evolution of Popular Music: USA 1960–2010." *Royal Society Open Science*. https://royalsocietypublishing.org/doi/10.1098/rsos.150081
  - Billboard Hot 100 feature evolution over 50 years. Three revolutions detected.

---

## Medium-weight precedent (useful context, not direct predecessors)

### Audio foundation models
- **CLAP — Contrastive Language-Audio Pretraining.** https://www.emergentmind.com/topics/contrastive-language-audio-pretraining-clap
  - Cross-modal audio-text embeddings. General audio, not music-specific.

- **MuQ-MuLan (Tencent AI Lab, January 2025).** 72.4% accuracy on perceptual music similarity.

- **Grötschla, F. et al. (2024).** "Towards Leveraging Contrastively Pretrained Neural Audio Embeddings for Music Similarity." ETH Zurich. *CEUR Workshop Proceedings*, Vol. 3787. https://ceur-ws.org/Vol-3787/paper2.pdf
  - CLAP embeddings tested on OLGA dataset for artist similarity. Effective and complementary to graph topology.

- **Van den Oord, A. et al. (2013).** "Deep Content-Based Music Recommendation." *NeurIPS 2013*. http://papers.neurips.cc/paper/5004-deep-content-based-music-recommendation.pdf
  - CNN predicting CF latent factors from mel-spectrograms. Landmark paper bridging audio and CF.

### Cross-modal / LLM-based recommendation (2024–2026 frontier)
- **CrossMuSim (Huawei, ICASSP 2025).** https://arxiv.org/html/2503.23128v1
  - GPT-4o-mini + Qwen2-7B generate text descriptions → contrastive learning trains audio embeddings. 15.57% improvement for Cantopop. Key finding: audio encoder internalizes artist-specific timbre even without artist names in training text.

- **De Nadai, M. et al. (Spotify, November 2025).** "Teaching Large Language Models to Speak Spotify: How Semantic IDs Enable Personalization." https://research.atspotify.com/2025/11/teaching-large-language-models-to-speak-spotify-how-semantic-ids-enable
  - Semantic IDs let LLMs refer to catalog items as tokens.

- **TalkPlay — Doh, S. & Nam, J. (ICML 2025).** "TalkPlay: Multimodal Music Recommendation with Large Language Models." https://arxiv.org/html/2502.13713v3
  - Music rec as next-token prediction.

- **FusID (January 2026).** "FusID: Modality-Fused Semantic IDs for Generative Music Recommendation." https://arxiv.org/html/2601.08764
  - 6.21% MRR improvement over TalkPlay.

- **JAM — Just Ask for Music (RecSys 2025).** https://dl.acm.org/doi/10.1145/3705328.3748020
  - User-query-item interactions as vector translations (TransE-inspired).

### GNN / knowledge graph approaches
- **Korzeniowski, F., Oramas, S., & Gouyon, F. (2022).** "Artist Similarity for Everyone: A Graph Neural Network Approach." *Transactions of ISMIR*. https://transactions.ismir.net/articles/10.5334/tismir.143
  - OLGA dataset: 17,673 artists, AllMusic ground truth, AcousticBrainz features. GNNs with triplet loss + connection dropout.

- **Music Galaxy — Primozic, C.** "Using Graph Embeddings for Music Visualization + Discovery with node2vec." https://cprimozic.net/blog/graph-embeddings-for-music-discovery/
  - 70,000+ artists in 3D via node2vec on Spotify Related Artists graph. Independent project.

### Production technique (marginal but interesting)
- **Koo, J. et al. (2022).** "Music Mixing Style Transfer: A Contrastive Learning Approach." *arXiv*. https://arxiv.org/pdf/2211.02247
  - FXencoder captures mixing style disentangled from musical content. Contrastive learning on mix parameters.

### Playlist co-occurrence
- **Spotify Million Playlist Dataset / RecSys 2018 Challenge.** https://research.atspotify.com/publications/recsys-challenge-2018-automatic-music-playlist-continuation
  - 1M playlists, 2M+ tracks. Word2Vec-style "playlist as sentence" embeddings.

### Social media mining
- **Schedl, M. (2010/2012).** "Mining Microblogs to Infer Music Artist Similarity and Cultural Listening Patterns." *WWW 2012*. https://dl.acm.org/doi/10.1145/2187980.2188218
  - Twitter #nowplaying co-occurrence for artist similarity. Also: https://www.cp.jku.at/people/schedl/Research/Publications/pdf/microblogging_ismir_2010.pdf

---

## Lower-weight (background reading, not direct precedent)

- **Hybrid GNN music rec (Springer, 2024).** PinSage evaluation. https://link.springer.com/article/10.1007/s11257-024-09410-4
- **Amazon Music transformer rec (OpenReview, 2025).** gSASRec for music. https://openreview.net/pdf?id=5EhE8euL9L
- **Lamere, P. (2008).** Last.fm tag analysis / folksonomy for music similarity.
- **Logan, B. et al. (2004).** Lyrics via PLSA on 40,000+ songs.
- **WASABI Song Corpus.** Metadata + NLP-processed lyrics (no liner notes).
- **Bogdanov, D. et al.** Combined timbral, tempo, semantic distances for hybrid similarity.
- **Content filtering methods review (arxiv, July 2025).** https://arxiv.org/html/2507.02282v1
- **Liner notes — Wikipedia.** https://en.wikipedia.org/wiki/Liner_notes (for context on the dataset gap)
- **RYM discovery guide.** https://rateyourmusic.com/wiki/RYM:How+to+Discover+Music+Using+RYM
- **Chartmetric AI genre comparison blog post.** https://hmc.chartmetric.com/using-ai-to-make-meaningful-comparisons-between-musicians/
- **Sequenced music rec with popularity awareness (arxiv, 2024).** https://arxiv.org/html/2409.04329v1
- **Social music discovery / friend-based rec (Springer, 2024).** https://link.springer.com/article/10.1007/s11042-024-19505-0
- **Knowledge graph + multi-task learning (Nature Scientific Reports, 2024).** https://www.nature.com/articles/s41598-024-52463-z
- **Visualizing 500 classical composers (ResearchGate, 2019).** https://www.researchgate.net/publication/334406188
