<!-- Text copy of docs/reference/HW1_proposal_EN.pdf (submitted HW1). For reference; docs/00_PROJECT_BRIEF.md has the updated plan. -->

## Learning Radio Shadows on the Moon: A Ray-Aligned, Physics-Supported Neural Network for Pathloss Map Prediction on Cratered Terrain

Abdurrahim Yılmaz (820220519) · EHB420E Artificial Neural Networks · Istanbul Technical University · October 29, 2026

## Abstract

NASA's Artemis program and the LunaNet plan make wireless communication on the Moon a near-term need. Pathloss maps show where a transmitter's signal reaches on cratered terrain and are the basis of network planning. Ray tracing can make these maps, but it is slow and must be repeated for every transmitter position. Neural networks for radio maps are well studied for cities on Earth, but lunar terrain is new: the first open dataset was released in 2026. On this dataset, the reference model RadioUNet has 4.66 dB error at 415 MHz and 6.28 dB at 5.8 GHz; sharp shadows behind crater rims are the hardest part. I think the main reason is that current models treat the map as a normal image and ignore that the signal travels along rays. I propose a neural network that converts the map to polar coordinates around the transmitter, so each ray becomes one column. It scans the highest obstacle along each ray and learns a correction to a knife-edge diffraction estimate. My hypothesis is that this ray-aligned processing reduces the 5.8 GHz error by at least 0.3 dB compared with the same network without it, with most of the gain in shadow regions. I will test this on 25,720 training maps per band with three runs and a region-based error analysis. The expected contribution is a ray-aligned, physics-supported model for lunar terrain, a measured analysis of where and why errors happen, and an open coverage planning demo.

## 1. Introduction and motivation

I am an electrical and electronics engineering student, and communication is the area I like most. While I was searching for a project with open data, I found a new lunar radio map dataset released in 2026. I chose this topic because it brings together communication, artificial intelligence and space.

This is a real need. NASA's LunaNet plans a communication and navigation network for landers, rovers and astronauts on the Moon [1]. In January 2025, Nokia integrated its 4G/LTE system into Intuitive Machines' IM-2 lander to build the first cellular network on the Moon [2].

To build such a network, the first question is: if I put a transmitter here, where will the signal reach? There are no buildings on the Moon, but craters, crater rims and hills block the signal. Pathloss maps answer this question. They are needed to choose antenna positions and to plan how far a rover can go without losing the link. Mission planners, companies and, in the end, astronauts are affected. The same idea may also help coverage planning in rural and mountain areas on Earth.

## 2. Problem statement

Precise problem: Given a 256×256 lunar height map (1 m per pixel) and the transmitter position, predict the pathloss map in dB at 415 MHz and 5.8 GHz [3]. The prediction must be accurate (low RMSE on valid pixels) and fast: the dataset asks for less than 500 ms per map, much faster than ray tracing [3]. The two bands come from the LunaNet specifications [4].

Evidence:

- Accurate maps come from ray tracing tools such as Sionna RT [5]. But the simulation must run again for every transmitter position, and its cost makes it impractical for most applications [6].

- Ray tracing cannot solve some lunar pixels: about 11.4% of the 5.8 GHz pixels are unresolved in the dataset [3], and RadioLunaDiff had to recompute maps and fill gaps by interpolation [4].

- Fast neural models are not accurate enough: the reference model RadioUNet has 4.66 dB error at 415 MHz and 6.28 dB at 5.8 GHz [3]. 6 dB is about four times in received power, which is enough to change a "covered / not covered" decision.

Why is it not solved yet? Almost all radio map studies are about cities; for example, the 2023 benchmark used 701 building maps from six cities [7]. The first lunar datasets appeared only in 2025-2026 [3], [4].

Root cause or symptom? The high 5.8 GHz error is a symptom; the organizers say the hard part is the sharp shadows from crater rims [3]. I think the root cause is the representation. Whether a pixel is in shadow depends on all the terrain on the line to the transmitter, which can be up to 360 pixels long. But standard CNNs look at small local windows on a Cartesian grid. This is my hypothesis, and I test it in Section 6.

## 3. Critical literature review

### 3.1 Search method

- Databases: IEEE Xplore, arXiv and Google Scholar, plus the dataset website and code.

- Keywords: "radio map prediction", "pathloss map deep learning", "RadioUNet", "lunar radio propagation", "lunar terrain path loss", "knife-edge diffraction deep learning", "physics-informed radio map", "polar transform CNN".

- Time range: 2019-2026 for deep learning; key older sources for diffraction (1966, ITU-R). Searched in October 2026.

- Inclusion: pathloss or radio map prediction with a clear method and numerical results; models and tools I will use.

- Exclusion: fingerprint localization only, no numerical evaluation, or no full text.

I used 21 sources; the table compares 13 of them. The last column is my own comment.

### 3.2 Comparison table

| Source | Data | Method | Strength | Limitation |
|---|---|---|---|---|
| [8] RadioUNet (2021) | City maps, ray tracing | U-Net CNN | Fast, open code; standard baseline | Treats the map as a normal image |
| [9] Thrane et al. (2020) | Real measurements, satellite images | Deep learning + simple pathloss model | Model help gives about 1 dB | Terrain geometry not modeled |
| [7], [6] 2023 benchmark | 701 city maps; Istanbul test maps | Common task, RMSE | Fair comparison | All methods worse on the test set |
| [10] PMNet (2023) | 2023 benchmark | CNN, augmentation, fine-tuning | Ranked first | City-specific; hard to interpret |
| [11] REM-U-Net (2024) | 2023 benchmark | U-Net; its LoS-input version had the second-lowest error | About 14 ms | Not tested on terrain |
| [12] RadioDiff (2025) | Dynamic city scenes | Diffusion model | Map as conditional generation | Heavy for 500 ms |
| [13] RMTransformer (2025) | Simulated data | Transformer encoder, CNN decoder | Over 30% lower RMSE than CNNs | Data not clear |
| [14] Indoor benchmark (2025) | 25 indoor scenes, 3 bands | All teams used physics inputs | Physics inputs are now standard | High error far from the transmitter |
| [15] Jaensch et al. (2026) | Indoor, noisy inputs | CNN + noise augmentation | 2.1 dB vs 3.1 dB for ray tracing | Single room; detailed materials did not help |
| [16] Chen and Chen (2024) | Two city datasets | Learning with multiple knife-edge diffraction | 10-18% better than RadioUNet and PMNet | Buildings only |
| [17] Hussain (2026) | 150 scenes, 1.8-28 GHz | Residual on a free-space + knife-edge prior | 35% lower RMSE at new frequencies | Cartesian grid |
| [4] RadioLunaDiff (2025) | Synthetic lunar terrain, 415 MHz and 5.8 GHz | Two U-Nets + diffusion | Beats RadioUNet and PMNet | Normalized metrics; no ray direction |
| [18] Adebowale and Ostermann (2026) | Real lunar terrain, 5.855 GHz | Ray tracing, path loss exponent | 5G band on real lunar terrain | Strong antenna effect (2.54 vs 4.33); no neural network |

### 3.3 Critical evaluation

What works? U-Net type models are fast (about 10-20 ms) and strong [7], [11], [14]. Physics inputs such as free-space loss, line of sight and diffraction appear in almost every good method [9], [11], [14], [16], [17]. Transformers and diffusion add detail but also cost [4], [12], [13].

Where do they fail? Generalization: in 2023 all methods were clearly worse on the test set [7], and indoor errors were high far from the transmitter [14]. On the Moon, this may mean the far shadow regions behind crater rims.

Assumptions and data limits: Models are trained on ray-tracing simulations, so they learn the simulator's assumptions. On the Moon, the terrains are also synthetic and there are no real measurements [4]. In [18], the path loss exponent changes from 2.54 to 4.33 only because of the antenna pattern, so even the "ground truth" depends on modeling choices.

Contradictions: Detailed material data was not better than a simple binary map [15], so more input is not always better; this is also a risk for my physics features. RadioLunaDiff gives only normalized metrics, so it cannot be compared directly with the dB RMSE of the dataset [3], [4].

Reproducibility: RadioUNet and the dataset code are open [3], [8], and the benchmarks use common test sets [7], [14]. Some preprints do not describe their data clearly [13].

Could my gap come from an incomplete search? I found no transmitter-centered polar or ray-wise architecture for radio maps; polar transforms are used in image classification [19]. I will search again before the final report.

## 4. Research gap

Gap (methodological): No neural network for lunar pathloss prediction builds the nature of shadowing into its architecture: shadowing starts at the transmitter and accumulates along each ray, so it is not a local effect.

Supporting publications: The only lunar neural network study, RadioLunaDiff, uses Cartesian U-Nets and diffusion [4], and the dataset's reference model is also a Cartesian U-Net [3], [8]. Physics-aware Earth studies add physics as an input or a prior but stay on a Cartesian grid [9], [16], [17], and line of sight is only an extra input channel [7], [11]. Polar transforms were used in vision for robustness to rotation [19], not for propagation.

Why does it matter? 5.8 GHz is the higher band, which usually carries higher data rates, and it has the largest baseline error [3]. A model that smooths shadows may mark an area behind a crater rim as covered, and a rover or an astronaut there would lose the link.

What if nobody solves it? Planners must choose between slow ray tracing for every candidate position and fast models that are wrong in shadows. The gap pixels of ray tracing also stay unsolved [3], [4].

## 5. Proposed solution

I call my model RayNet. The network looks at the map from the transmitter's point of view and reads each ray in order, from the transmitter outward.

### 5.1 Mechanism

- Physics prior: For each pixel, I compute the free-space loss plus a single knife-edge loss for the strongest obstacle on the terrain profile: ν = h·√(2(d1+d2) / (λ·d1·d2)) and J(ν) ≈ 6.9 + 20·log10(√((ν−0.1)²+1) + ν − 0.1) dB for ν > −0.78 [20]. Here h is the obstacle height above the ray, d1 and d2 are its distances to the two ends, and λ is the wavelength. For several obstacles, I will try the Deygout method [21]. This step needs no training.

- Polar transform: I resample the height map and the prior to an (r, θ) grid around the transmitter, for example 256 × 360. Then each column is one ray.

- Ray scan: On each ray, a cumulative maximum gives "the largest elevation angle seen so far". How far a pixel is below this horizon is its shadow depth. This differentiable layer is followed by 1D convolutions along rays and 2D convolutions across neighboring rays, so the network can learn how diffraction spreads sideways.

- Back transform and correction: The output goes back to the Cartesian grid, and a small U-Net adds a correction: prediction = physics prior + learned correction.

- Training: masked MSE on valid pixels [3]; I will compare per-band models with one frequency-conditioned model.

### 5.2 Scientific reasoning

Shadowing is decided by the terrain profile between the transmitter and the receiver, which is exactly what knife-edge models use [20], [21]. The polar transform puts this profile into columns, so the network's view follows the propagation direction and rotations become simple shifts [19]. With the prior, the network only learns the error of the physics model; model-aided and residual learning like this helped on Earth [9], [17].

### 5.3 Originality, alternatives and weakest assumption

Is it new? It is a combination of known ideas (U-Net, physics priors and polar transforms [8], [17], [19]) in a new problem. My part is the ray-wise causal scan on a transmitter-centered polar grid, its test on the first open lunar dataset, and a region-by-region error analysis.

Simpler alternatives: line-of-sight and free-space channels added to RadioUNet [11], [14]; a larger Cartesian network such as PMNet or a transformer [10], [13]; the ITU-R model alone [20]; and diffusion refinement [4], which is probably too slow for 500 ms. I will test the first three. If one of them is as good as RayNet, my method is not needed, and I will report this.

Weakest assumption: that most of the error comes from long-distance blocking and shadow edges. It may also come from crater-wall reflections, the area near the transmitter or mask edges. So in weeks 1-2, I will first split the baseline error by region, and I will change the method if shadows are not the main source.

## 6. Methodology and validation

### 6.1 Data

I will use the open lunar dataset of the ICASSP 2027 lunar radio map challenge, which is available on Kaggle [3]. It has 66,000 pathloss maps at 415 MHz and 5.8 GHz (5.775 GHz). The training split has 6,120 terrains (25,720 maps per band), the frozen validation split has 680 terrains (2,330 maps per band), and the test labels are hidden [3]. The splits are made by terrain. Each 256×256 map (1 m per pixel) has a validity mask, and unresolved pixels are not scored [3].

### 6.2 Experiments and baselines

| Code | Experiment | Purpose |
|---|---|---|
| E0 | RadioUNet baseline | Reproduce 5.47 dB [3] |
| E1 | Physics prior only | Lower bound without learning |
| E2 | RadioUNet + LoS and free-space channels | Simple alternative |
| E3 | E2 + residual on the physics prior | Effect of the prior |
| E4 | RayNet (E3 + polar ray module) | Main proposal |
| E5 | Ablations: no scan layer; parameter-matched Cartesian model; one vs per-band models | Source of the gain |
| E6 | Test-time augmentation | Extra gain, reported separately |

### 6.3 Metrics

- Main metric: RMSE in dB on valid pixels, per band and 50/50 combined [3].

- Error analysis: RMSE for line-of-sight (LoS) and non-line-of-sight (NLOS) pixels, for a 3-pixel "border band" around the LoS/NLOS edge, and by distance.

- Coverage accuracy: The received power is 20 dBm minus the pathloss [3]. For a −100 dBm threshold (120 dB pathloss), I measure how often the "covered / not covered" label is correct.

- Speed: inference time per map, under 500 ms [3].

### 6.4 Hypotheses and success criteria

- H1: With the same backbone and budget, RayNet (E4) lowers the 5.8 GHz RMSE by at least 0.3 dB compared with E3 (mean of three seeds; paired Wilcoxon test over maps, p < 0.05).

- H2: At least half of the gain (in squared error) comes from NLOS and border pixels.

- H3: RayNet's combined RMSE is at least 0.5 dB below the 5.47 dB baseline, with inference under 500 ms.

Sample size: There are 2,330 validation maps per band. Even with a 1-2 dB standard deviation of the per-map differences, the standard error of the mean is about 0.02-0.04 dB, which is enough to detect 0.3 dB. Three seeds show the variation from training.

What would disprove my idea? If the H1 gain is under 0.3 dB or not significant, or if it is the same in LoS and NLOS pixels, my explanation is wrong.

### 6.5 Bias, overfitting and alternative explanations

- Overfitting: I will select models with my own split of the training terrains and report final results on the frozen validation split, as the dataset rules ask [3]. I will use early stopping and report the mean and standard deviation of three seeds. A hidden-test score is only possible through the challenge server; this is optional and not needed for the course.

- Alternative explanations: more parameters (E5 uses a parameter-matched model), physics inputs (E2 and E3 are measured separately) and test-time augmentation (E6 is reported separately).

- Simulator bias: All labels come from Sionna RT on synthetic terrain [4], [5], so the results may not hold on the real Moon. I will report this as a limitation.

- Reproducibility: fixed seeds, configuration files and open code on GitHub.

Negative result: Even if H1 fails, the region-based analysis will show whether shadows or reflections are the main difficulty.

### 6.6 Timeline

Weeks 1-2: data check, E0 and baseline error analysis. Weeks 3-4: E1-E3. Weeks 5-6: E4-E6. Weeks 7-8: final results, report and demo.

## 7. Feasibility and limitations

### 7.1 Resources

| Resource | What I need | Status |
|---|---|---|
| Data | Open dataset on Kaggle [3] | Free |
| Hardware | Laptop, Kaggle's free GPU quota, Colab Pro (200 units, about 165 T4 hours) | Enough; I estimate 20-40 GPU hours |
| Software | Python, PyTorch, NumPy; the dataset's loader and reference code [3] | Open source |
| Skills | Communication courses and this course | I need to learn polar resampling and ITU-R diffraction [20] |
| Time and budget | About 8 weeks; no extra cost | Fits the semester |

### 7.2 Ethics, privacy, safety and regulation

There is no personal data. The data is synthetic; I will follow its terms of use and cite it [3]. The model must not be used alone for mission-critical decisions: final decisions need ray tracing and measurements, and the demo will say this clearly.

### 7.3 Risks and actions

- Polar distortion near the transmitter: I will give more weight to the Cartesian correction there.

- Large data (about 10 GB, my estimate): I will read it with memory mapping.

- No gain over simple alternatives: I will still report the results and the error analysis.

- GPU quota: I will share the work between Kaggle and Colab and start with one band.

### 7.4 Realistic scope

I can reproduce the baseline, test RayNet and the ablations, analyze the errors and build a demo. Beyond my capacity are real lunar measurements, uncertain regolith properties, multi-transmitter optimization and different antenna patterns, which can change pathloss a lot [18].

## 8. Expected results and impact

Testable outputs: a reproduced baseline (about 5.47 dB); RayNet with open code; a results table for E0-E6; an error analysis for LoS, NLOS and border regions; and a demo that shows the coverage map right after a click.

Who benefits? Lunar mission planners and companies that build lunar networks [1], [2] can test many antenna positions and rover routes in seconds. Radio map researchers get an open comparison for the Moon. Later, the same idea may help coverage planning in rural and mountain areas on Earth.

Unintended harms: People may trust a fast model too much and forget that it was trained on synthetic terrain, and a wrong prediction can cause link loss. So the demo will warn about border regions, where the errors are highest.

Next research question: Does a model trained on synthetic terrain generalize to real lunar height maps [18] and to rural terrain on Earth? Can such a fast model choose the positions of several transmitters inside an optimization loop?

## 9. Researcher's reflection

1. The Skeptic: An expert says this research is unnecessary. What is my strongest evidence-based answer? The likely objection is "ray tracing already exists". But ray tracing must run again for every transmitter position and is too costly for most applications [6]. The dataset itself asks for under 500 ms per map [3]. Ray tracing also leaves about 11.4% of the 5.8 GHz pixels unresolved, and the current fast model has 6.28 dB error, about four times in received power [3]. Real networks are already going to the Moon [1], [2], so a tool that is both fast and accurate is a real need.

2. The Failure: After a year, my hypothesis is wrong. What valuable knowledge remains? An open pipeline that reproduces the baseline on the first open lunar dataset; an error analysis for LoS, NLOS and border regions that shows what is really hard; the physics prior (E1) as a cheap baseline; and the useful result that explicit ray processing is not needed.

3. The Originality: If I remove everything borrowed from prior work, what is my own contribution? U-Net [8], residual learning on a physics prior [17], polar transforms [19] and the dataset [3] are not mine. My contribution is the ray-wise "horizon scan" on a transmitter-centered polar grid inside a radio map network, the test of whether it helps NLOS and border pixels on lunar terrain (H1, H2), and the region-based error and coverage analysis. It is modest, but clear and measurable.

4. The Reality: With €500, one computer and three months, what is the minimum meaningful experiment? I would use only the 5.8 GHz band: reproduce RadioUNet with Kaggle's free GPU (E0), compute the prior on my laptop (E1), and train E3 and E4 with three seeds each. Comparing the NLOS and border RMSE is enough to test H1 and H2. This needs about 10-20 GPU hours, costs between zero and €50, and takes about four weeks.

5. The Intellectual Honesty: What is the strongest argument against my idea, and what evidence would change my mind? A large Cartesian network with horizon and line-of-sight input channels may learn the radial structure by itself, and polar resampling may add interpolation errors that cancel the gain. Synthetic terrain may also differ from the real Moon. If a parameter-matched Cartesian model with the same horizon features (E5) reaches RayNet's NLOS error within one standard deviation over three seeds, I will accept that explicit ray processing is not needed and recommend the simpler model.

## 10. Use of AI tools

I chose this topic after a long search for project ideas, and I want to be open about how I used AI. I used an AI assistant (Claude, by Anthropic) as a research assistant, not as a decision maker: it helped me find and summarize papers and datasets, clarify my plan and my contribution, and write and edit this English text. The idea of my model grew out of the Earth-based radio map studies in Section 3, especially physics-based inputs and line-of-sight features. All references were checked (authors, venue, year and DOI) on publisher pages, arXiv or official records. Established findings are cited, and my own ideas are marked as "I think" or "my hypothesis". The decisions and all claims in this proposal are my responsibility.

## References

[1] D. J. Israel, K. D. Mauldin, C. J. Roberts, J. W. Mitchell, A. A. Pulkkinen, L. D. Cooper, M. A. Johnson, S. D. Christe, and C. J. Gramling, "LunaNet: a flexible and extensible lunar exploration communications and navigation infrastructure," in Proc. IEEE Aerospace Conference, Big Sky, MT, USA, 2020. NASA NTRS 20200001555

[2] Nokia, "Nokia's cellular network ready for Moon as Intuitive Machines completes final lunar lander installation," press release, Jan. 7, 2025. nokia.com

[3] The First Lunar Pathloss Radio Map Prediction Challenge (IEEE ICASSP 2027 SP Grand Challenge), website and "Dataset & Instructions" page, lunarradiomapchallenge.github.io; data on Kaggle (accessed Oct. 29, 2026).

[4] P. Torrado, A. Pearson, J. Klein, A. Moscibroda, and J. Smith, "RadioLunaDiff: Estimation of wireless network signal strength in lunar terrain," arXiv:2509.14559, 2025.

[5] J. Hoydis, F. Aït Aoudia, S. Cammerer, M. Nimier-David, N. Binder, G. Marcus, and A. Keller, "Sionna RT: Differentiable ray tracing for radio propagation modeling," in Proc. IEEE Globecom Workshops (GC Wkshps), 2023.

[6] Ç. Yapar, F. Jaensch, R. Levie, G. Kutyniok, and G. Caire, "Overview of the first pathloss radio map prediction challenge," IEEE Open J. Signal Process., vol. 5, pp. 948-963, 2024, doi:10.1109/OJSP.2024.3419563.

[7] Ç. Yapar, F. Jaensch, R. Levie, G. Kutyniok, and G. Caire, "The first pathloss radio map prediction challenge," arXiv:2310.07658, 2023 (ICASSP 2023 Grand Challenge).

[8] R. Levie, Ç. Yapar, G. Kutyniok, and G. Caire, "RadioUNet: Fast radio map estimation with convolutional neural networks," IEEE Trans. Wireless Commun., vol. 20, no. 6, pp. 4001-4015, 2021, doi:10.1109/TWC.2021.3054977.

[9] J. Thrane, D. Zibar, and H. L. Christiansen, "Model-aided deep learning method for path loss prediction in mobile communication systems at 2.6 GHz," IEEE Access, vol. 8, pp. 7925-7936, 2020, doi:10.1109/ACCESS.2020.2964103.

[10] J. H. Lee, J. Lee, S.-H. Lee, and A. F. Molisch, "PMNet: Large-scale channel prediction system for ICASSP 2023 first pathloss radio map prediction challenge," in Proc. IEEE ICASSP, 2023, pp. 1-2, doi:10.1109/ICASSP49357.2023.10095257.

[11] H. Sallouha, S. Sarkar, E. Krijestorac, and D. Cabric, "REM-U-Net: Deep learning based agile REM prediction with energy-efficient cell-free use case," IEEE Open J. Signal Process., vol. 5, pp. 750-765, 2024, doi:10.1109/OJSP.2024.3378591.

[12] X. Wang, K. Tao, N. Cheng, Z. Yin, Z. Li, Y. Zhang, and X. Shen, "RadioDiff: An effective generative diffusion model for sampling-free dynamic radio map construction," IEEE Trans. Cogn. Commun. Netw., vol. 11, no. 2, pp. 738-750, 2025, doi:10.1109/TCCN.2024.3504489.

[13] Y. Li, C. Zhang, W. Wang, and Y. Huang, "RMTransformer: Accurate radio map construction and coverage prediction," arXiv:2501.05190, 2025.

[14] S. Bakirtzis, Ç. Yapar, K. Qiu, I. Wassell, and J. Zhang, "The first indoor pathloss radio map prediction challenge," arXiv:2501.13698, 2025 (ICASSP 2025).

[15] F. Jaensch, Ç. Yapar, G. Caire, and B. Demir, "Radio map prediction from noisy environment information and sparse observations," arXiv:2602.11950, 2026.

[16] W. Chen and J. Chen, "Diffraction and scattering aware radio map and environment reconstruction using geometry model-assisted deep learning," arXiv:2403.00229, 2024.

[17] S. Hussain, "Zero-shot frequency generalization for radio map prediction via cross-attention physics-residual learning," arXiv:2608.16194, 2026.

[18] Q. R. Adebowale and S. Ostermann, "Sub-6-GHz 5G large-scale path loss model for Shoemaker Rim F: Sensitivity to transmitter antenna pattern," Telecom, vol. 7, no. 1, art. 21, 2026, doi:10.3390/telecom7010021.

[19] C. Esteves, C. Allen-Blanchette, X. Zhou, and K. Daniilidis, "Polar transformer networks," in Proc. Int. Conf. Learning Representations (ICLR), 2018.

[20] ITU-R, "Recommendation ITU-R P.526-16: Propagation by diffraction," International Telecommunication Union, Nov. 2025.

[21] J. Deygout, "Multiple knife-edge diffraction of microwaves," IEEE Trans. Antennas Propag., vol. 14, no. 4, pp. 480-489, 1966, doi:10.1109/TAP.1966.1138719.
