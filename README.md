<h1>Botsheild: Bot Detection</h1>

<p>
  A bot and fake account detection system for social media, built on data mining primitives and a Random Forest classifier.
  No deep learning. No black box. The system can explain exactly why an account was flagged.
</p>

<p>
  Built on the <a href="https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts">Cresci-2017</a> dataset.
</p>

<hr>

<h2>What this does</h2>

<p>
  Most bot detectors train a model, hand it thousands of features, and return a probability score.
  This project takes a different path. It treats every account's following list as a set,
  finds accounts that follow suspiciously similar people using locality-sensitive hashing,
  mines behavioral rules from those clusters, and only then runs a classifier on a compact, meaningful feature vector.
</p>

<p>
  The result is a layered pipeline where each stage has a clear job and can be inspected independently.
</p>

<hr>

<h2>Pipeline overview</h2>

<pre>
Raw account data (Cresci-2017)
        |
        v
MinHash + LSH  (datasketch)
  - Compress each account's following list into a MinHash signature
  - Use LSH to bucket accounts by signature similarity
  - Output: candidate clusters of structurally similar accounts
        |
        v
Association Rule Mining  (mlxtend)
  - Treat following lists as market baskets
  - Run Apriori to find frequent itemsets
  - Extract rules: {follows @X, follows @Y} -> likely bot
  - Output: interpretable behavioral rules + per-account rule confidence scores
        |
        v
Feature Engineering
  - Jaccard similarity score with nearest bot cluster
  - Association rule confidence score for this account
  - Bloom Filter hit flag (known bot fingerprint match)
  - Follower / following ratio
  - Account age in days
  - Average tweets per day
  - Default profile picture flag (binary)
        |
        v
Random Forest Classifier  (scikit-learn)
  - Trained on labeled Cresci-2017 ground truth
  - Outputs bot probability score per account
  - Feature importance ranking included
        |
        v
Bloom Filter  (pybloom-live)
  - Populated with fingerprints of confirmed bots
  - Used for real-time screening of new unknown accounts
  - O(1) lookup, configurable false positive rate
</pre>

<hr>

<h2>Tech stack</h2>

<table border="1" cellpadding="6" cellspacing="0">
  <tr>
    <th>Purpose</th>
    <th>Library</th>
    <th>Why</th>
  </tr>
  <tr>
    <td>Data loading and cleaning</td>
    <td>pandas</td>
    <td>Standard. Nothing to argue about.</td>
  </tr>
  <tr>
    <td>MinHash and LSH</td>
    <td>datasketch</td>
    <td>Has MinHash and MinHashLSH built in. Saves writing hash permutations manually.</td>
  </tr>
  <tr>
    <td>Association rule mining</td>
    <td>mlxtend</td>
    <td>apriori() and association_rules() work directly on transaction lists.</td>
  </tr>
  <tr>
    <td>Bloom Filter</td>
    <td>pybloom-live</td>
    <td>One import, configurable FP rate, done.</td>
  </tr>
  <tr>
    <td>ML classifier</td>
    <td>scikit-learn (RandomForestClassifier)</td>
    <td>Feature importances out of the box. Works well on mixed numerical and binary features.</td>
  </tr>
  <tr>
    <td>Evaluation</td>
    <td>scikit-learn metrics</td>
    <td>classification_report gives precision, recall, F1 in one call.</td>
  </tr>
  <tr>
    <td>Visualization</td>
    <td>matplotlib, seaborn</td>
    <td>Confusion matrix, feature importance bar chart.</td>
  </tr>
  <tr>
    <td>Explainability (optional)</td>
    <td>shap</td>
    <td>SHAP values on the Random Forest to explain individual predictions.</td>
  </tr>
</table>

<hr>

<h2>Dataset</h2>

<p>
  <a href="https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts">Cresci-2017</a> from Kaggle.
  Labeled, clean, and widely used in academic bot detection work.
  Contains genuine accounts and multiple categories of bots (social spambots, traditional spambots, fake followers).
</p>

<p>
  Download and place the CSV files inside <code>data/raw/</code> before running anything.
</p>

<hr>

<h2>Project structure</h2>

<pre>
bot-detection/
  data/
    raw/              # Cresci-2017 CSVs go here (not committed)
    processed/        # Cleaned and feature-engineered outputs
  src/
    ingest.py         # Load and clean raw data
    minhash_lsh.py    # Build MinHash signatures and LSH index
    association.py    # Apriori and rule extraction
    features.py       # Assemble final feature matrix
    classifier.py     # Train and evaluate Random Forest
    bloom.py          # Build and query Bloom Filter
    evaluate.py       # Metrics and plots
  notebooks/
    exploration.ipynb # Initial data exploration
    full_pipeline.ipynb
  requirements.txt
  README.html
</pre>

<hr>

<h2>Installation</h2>

<pre>
git clone https://github.com/your-username/bot-detection.git
cd bot-detection
pip install -r requirements.txt
</pre>

<p>requirements.txt contents:</p>

<pre>
pandas
numpy
datasketch
mlxtend
pybloom-live
scikit-learn
matplotlib
seaborn
shap
jupyter
</pre>

<hr>

<h2>Running the pipeline</h2>

<p>Either run each module in order:</p>

<pre>
python src/ingest.py
python src/minhash_lsh.py
python src/association.py
python src/features.py
python src/classifier.py
python src/bloom.py
python src/evaluate.py
</pre>

<p>Or open <code>notebooks/full_pipeline.ipynb</code> to run everything interactively with outputs visible at each step.</p>

<hr>

<h2>How the core pieces work</h2>

<h3>MinHash and LSH</h3>

<p>
  Every account has a following list. That list is treated as a set.
  A MinHash signature compresses that set into a fixed-length vector of hash values while approximately preserving Jaccard similarity between accounts.
  LSH then splits signatures into bands and hashes each band into a bucket.
  Two accounts that land in the same bucket for any band are flagged as a candidate pair.
</p>

<p>
  The threshold is controlled by the number of bands (b) and rows per band (r), where total signature length k = b * r.
  More bands means a lower similarity threshold, more candidates, higher recall.
  Fewer bands means a stricter threshold, fewer candidates, higher precision.
  Tune these based on your false positive tolerance.
</p>

<h3>Association Rule Mining</h3>

<p>
  Each account's following list is treated as a transaction (a basket of items).
  Apriori finds itemsets that appear frequently across bot accounts.
  Rules of the form {follows @A, follows @B} -> bot are extracted above minimum support and confidence thresholds.
  The confidence score of the best matching rule becomes a feature for the classifier.
</p>

<h3>Random Forest</h3>

<p>
  The feature matrix combines signals from data mining (Jaccard score, rule confidence, Bloom Filter hit)
  with raw account metadata (ratio of followers to following, account age, tweet rate, default picture flag).
  RandomForestClassifier from scikit-learn is trained on the labeled Cresci ground truth.
  Feature importances show which signals contributed most to classification.
</p>

<h3>Bloom Filter</h3>

<p>
  After training, confirmed bot accounts have their following fingerprints inserted into a Bloom Filter.
  New unknown accounts can be screened against this filter in constant time.
  False positives are possible (configurable), false negatives are not.
  Useful as a lightweight pre-filter before running the full pipeline on new accounts.
</p>

<hr>

<h2>Evaluation metrics</h2>

<ul>
  <li>Precision, Recall, F1 score from the Random Forest on the Cresci test split</li>
  <li>Confusion matrix</li>
  <li>LSH candidate reduction ratio: how many accounts were pruned before the classifier ran</li>
  <li>Bloom Filter false positive rate using the formula P(FP) = (1 - e^(-kn/m))^k</li>
  <li>Feature importance bar chart from the Random Forest</li>
</ul>

<hr>

<h2>Key design decisions</h2>

<p>
  The data mining stage runs first deliberately.
  Running a classifier on 100,000 raw accounts directly is expensive and noisy.
  LSH reduces that to a much smaller set of structurally suspicious candidates first.
  Association rules then give the classifier meaningful features rather than making it figure out behavioral patterns on its own.
  Each stage has a clear and inspectable job.
</p>

<p>
  Random Forest was chosen over XGBoost because feature importances are native and intuitive,
  it is harder to overfit on this feature set, and the performance difference on a dataset this size is marginal.
  SHAP is included as an optional layer if you want per-account explanations beyond global feature importance.
</p>

<hr>

<h2>Limitations</h2>

<p>
  This system is built on following-list structure. Accounts that behave like bots but have randomized following lists
  will be harder to catch with LSH alone. The classifier's metadata features (account age, tweet rate) partially cover this gap,
  but sophisticated bots that mimic human posting behavior are out of scope here.
</p>

<p>
  The Bloom Filter only knows about bots it has seen before. It is not a substitute for the full pipeline on novel accounts.
</p>

<hr>

<h2>References</h2>

<ul>
  <li>Cresci, S. et al. (2017). The Rise of Social Bots. Communications of the ACM.</li>
  <li>Rajaraman, A. and Ullman, J.D. Mining of Massive Datasets. Chapter 3 (LSH) and Chapter 6 (Frequent Itemsets).</li>
  <li>datasketch documentation: <a href="https://ekzhu.com/datasketch/index.html">ekzhu.com/datasketch</a></li>
  <li>mlxtend documentation: <a href="https://rasbt.github.io/mlxtend/">rasbt.github.io/mlxtend</a></li>
</ul>

<hr>

<p>Built for a data mining course project. Feedback and issues welcome.</p>

</body>
</html>
