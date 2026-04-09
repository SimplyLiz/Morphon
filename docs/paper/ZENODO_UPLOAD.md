# Zenodo Upload Guide

Zwei Wege. Variante A ist schneller (5 Min, sofort fertig). Variante B ist sauberer langfristig (automatisches Archivieren bei jedem Release).

---

## Variante A: Direkter PDF-Upload (5 Minuten)

1. Konto anlegen auf https://zenodo.org/signup
   - Login mit GitHub geht auch (empfohlen — verbindet automatisch dein GH-Profil)

2. https://zenodo.org/uploads/new öffnen

3. **Upload files**: `docs/paper/paper/Morphogenic_Intelligence.pdf` hochladen

4. **Basic information** ausfüllen:
   - **Resource type**: `Publication > Preprint`
   - **Title**: `Morphogenic Intelligence: Runtime Neural Development Beyond Static Architectures`
   - **Publication date**: 2026-04-08
   - **Creators**:
     - `Welsch, Lisa` (kein ORCID nötig wenn du keinen hast)
     - `Kwiecień, Martyna`
   - **Description**: siehe Block unten — kopier den ganzen `Description`-Text
   - **License**: `Apache License 2.0` (passt zum Code-Repo)
   - **Language**: English

5. **Recommended** (optional aber empfohlen):
   - **Keywords**: morphogenic intelligence, spiking neural networks, structural plasticity, three-factor learning, no backpropagation, developmental neuroscience, self-healing networks, STDP, Rust, CartPole, MNIST
   - **Related identifiers**:
     - `URL: https://github.com/SimplyLiz/Morphon` — relation: `is supplemented by` — resource type: `Software`

6. **Publish** klicken
   - Du bekommst sofort einen DOI im Format `10.5281/zenodo.XXXXXXX`
   - Der DOI ist permanent und zitierbar
   - Datei ist sofort live unter `https://zenodo.org/records/XXXXXXX`

7. **Versionierung**: Wenn ihr später eine neue Version hochladet, klickt auf den Eintrag → "New version" → upload neues PDF → publish. Beide Versionen bleiben unter ihrem eigenen DOI zitierbar, plus es gibt einen "concept DOI" der immer auf die neueste Version zeigt.

---

## Variante B: GitHub-Integration (für automatische Archivierung)

Empfohlen wenn ihr plant, mehrere Versionen zu releasen.

1. Auf https://zenodo.org/account/settings/github/ einloggen
2. Repository `SimplyLiz/Morphon` aktivieren (Schalter umlegen)
3. Im GitHub-Repo einen Release erstellen:
   ```bash
   cd /Users/lisa/Work/Projects/Morphon
   git tag -a v3.0.0-paper -m "Morphogenic Intelligence preprint v1"
   git push origin v3.0.0-paper
   ```
   Dann auf GitHub: Releases → Create new release → Tag `v3.0.0-paper` → Title "Paper v1" → Publish
4. Zenodo bemerkt den Release automatisch und archiviert das gesamte Repo (inkl. PDF, LaTeX-Source, Code, Daten) als ein Bundle mit DOI
5. Die `.zenodo.json` Datei im Repo-Root (die ich gerade erstellt habe) liefert automatisch alle Metadaten

**Vorteil B**: Bei jedem zukünftigen Release entsteht automatisch eine neue archivierte Version mit eigenem DOI. Code, Daten und Paper sind alle im selben Snapshot.

**Nachteil B**: Der Download ist ein Repo-Tarball, nicht das nackte PDF — Leute die nur das Paper lesen wollen, müssen auspacken.

---

## Empfehlung

**Mach beide.** 

- **Variante A heute** für sofortigen, sauberen PDF-DOI den ihr in E-Mails / Tweets / arXiv-Endorsement-Anfragen nutzen könnt
- **Variante B nebenher** als Code+Paper-Bundle, das automatisch mit jedem Release mitwächst

Die Zenodo-DOIs sind unabhängig voneinander; ihr habt am Ende zwei zitierbare Einträge:
1. Der Paper-DOI (Variante A) — was Reviewer/Endorser zitieren
2. Der Software/Bundle-DOI (Variante B) — was Code-User zitieren

---

## Description text (zum Reinkopieren)

Wenn ihr Variante A nehmt, kopiert das hier ins Description-Feld auf zenodo.org. HTML wird unterstützt.

```html
<p>We introduce <strong>Morphogenic Intelligence</strong> (MI), a neural architecture in which compute units &mdash; called <em>Morphons</em> &mdash; grow, differentiate, fuse, and undergo apoptosis at runtime. Unlike Transformers, CNNs, RNNs, and SSMs, where topology is fixed at design time and only weights are learned, MI treats network architecture itself as a primary learned artifact.</p>

<p>Each Morphon carries a developmental program inspired by biological morphogenesis: chemical gradients in a hyperbolic embedding space guide connectivity, neuromodulatory signals gate plasticity, and a dual-clock scheduler separates fast inference from slow structural remodeling. We implement the full MI engine in Rust and evaluate it on CartPole control and MNIST classification.</p>

<p><strong>Headline results:</strong></p>
<ul>
  <li>CartPole-v1 is solved (avg=195.2 on the v0.5.0 codebase) using only local three-factor learning and developmental morphogenesis &mdash; no backpropagation through the spike network.</li>
  <li>On MNIST, the system reaches 44.5% intact accuracy in v4.1.0 (up from 30% in v3.0.0) and <strong>48% after destroying 30% of associative neurons and allowing the network to regrow</strong> &mdash; a self-healing response that exceeds the intact baseline.</li>
  <li>Morphon potentials carry discriminative representations that the spike pipeline cannot deliver intact to motor morphons; the analog readout over potentials is the correct output mechanism for classification in a spiking developmental architecture.</li>
</ul>

<p>We document and fix four characteristic failure modes (modulatory explosion, motor silencing, LTD vicious cycle, premature regulatory throttling), each with a direct biological analogue. The premature throttling bug is novel and broadly applicable to any biologically-inspired regulator that uses raw reward as a maturity signal.</p>

<p>The most striking finding is that beneficial self-healing exists: a network that has lost 30% of its hidden layer can regrow into a configuration that outperforms its undamaged predecessor, because the regrowth happens under the high-plasticity regime that the homeostatic regulator restored in response to the damage.</p>

<p>The MI engine, all experiments, the failure-mode catalogue, and the LaTeX paper source are available at <a href="https://github.com/SimplyLiz/Morphon">github.com/SimplyLiz/Morphon</a>.</p>
```

---

## Nach dem Upload

Du bekommst eine Zitations-Box wie:

```
Welsch, L., & Kwiecień, M. (2026). Morphogenic Intelligence: Runtime Neural
Development Beyond Static Architectures. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

BibTeX-Variante:
```bibtex
@misc{welsch2026morphogenic,
  author       = {Welsch, Lisa and Kwiecień, Martyna},
  title        = {Morphogenic Intelligence: Runtime Neural Development Beyond Static Architectures},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

Diesen DOI dann im README.md der Repo eintragen — ich kann das eintragen sobald du ihn hast.

---

## Endorsement-Mail-Vorlage für arXiv (separat)

Wenn ihr parallel auf arXiv kommen wollt, hier eine kurze Mail-Vorlage. Schickt sie an einen der Forscher die ihr im Paper zitiert:

> Subject: arXiv endorsement request — cs.NE preprint on developmental spiking architecture
>
> Dear Prof. [Name],
>
> I'm writing to ask if you'd be willing to endorse our preprint for arXiv (cs.NE category). We cite your work on [specific paper] in our manuscript.
>
> The paper introduces Morphogenic Intelligence: a Rust-implemented spiking architecture in which compute units grow, differentiate, fuse, and undergo apoptosis at runtime under local developmental rules. We solve CartPole-v1 with three-factor learning only (no backprop), document a novel "premature Mature" failure mode in homeostatic regulators on dense-reward tasks, and report a self-healing result on MNIST where post-damage recovery (+18pp) exceeds intact training.
>
> The preprint is available on Zenodo: [DOI]
> Code, experiments, and LaTeX source: https://github.com/SimplyLiz/Morphon
>
> Endorsement instructions: https://arxiv.org/auth/endorse?x=XXXXXX [diesen Link bekommst du beim ersten arXiv-submit-Versuch]
>
> Happy to answer any questions about the work.
>
> Best regards,
> Lisa Welsch & Martyna Kwiecień

**Gute Endorser-Kandidaten** (basierend auf wem ihr im Paper zitiert):
- **Tim Vogels** (iSTDP, Vogels et al. 2011) — Oxford
- **Wulfram Gerstner** (three-factor learning, Frémaux & Gerstner 2016) — EPFL
- **Sebastian Risi** (NDP, Najarro et al. 2023) — IT University Copenhagen
- **Friedemann Zenke** (spiking deep learning) — FMI Basel
- **Brett Kagan** (DishBrain) — Cortical Labs

Frag 2-3 parallel an, höflich und kurz. Erwartete Antwortrate ~30-50%.
