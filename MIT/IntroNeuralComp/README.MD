# Introduction to Neural Computation
## Prof. Michale Fee
## Daniel Zysman

# Overview and Ionic Currents
- Neuroscience now is about understanding and developing Engineering Level descriptions of Brain Sytems and "Circuits".
- There have been tremendous improvements in the technology to image and track changes in the brain.
- What we're after is how to relate this information to the "Circuit Models" which are being developed.
- Here is an example probe:
![Neuropixels Probes Carandini](images/neuropixel-probe-image.png)
- How do you relate this high dimensional datasets and use them?
- You use Mathematical Techniques to analyze meural data in a way that allows us to relate it to mathematical models.
- Song birds learn by imitating their parents songs.
- The course goals are:
  1. Understand the basic biophysics of neurons and networks and other pricniples underlying brain and cognitive functions.
  2. Use mathematical techniques to:
    * Analyze simple models of neurons and networks.
    * Do data analysis of behavioral and neuronal data.
  3. Become proficient at using numerical methods to implement these techniques.
![Course Topics](images/topics-of-the-course.png)
- We're going to start with a model of Neuron:
![Circuit Model of a Neuron](images/circuit-model-of-a-neuron.png)
- We do this since it's important that the current, voltage, resistance is well defined for the model of the brain.
![Cluster Diagram of Neuron Types](images/cluster-diagram-of-neuron-type.png)
- Neurons also have complex morphologies.
- If combine a capacitor with a resistor then it looks like a filter.
- For a circuit, you need wires and we therefore our *Circuit Model of the Brain* needs them too.
- This is the intra and extra cellular solution which is our wires.
- This area is filled with a salt solution.
- The first thing we need to discuss will be **Thermal Energy**.
![Velocity of Sodium Ion at Equilibrium](images/velocity-calculation-for-sodium-ion.png)
![Random Walking Partcle](images/particle-collisions-random-walk.png)
- Counter intuitively, **Diffusion** is fast at short length scales and slow at long length scales.
![Diffucsion Time lapse](images/diffusion-inside-human-body.png)
- It takes a very long time for the distribution of particles to flatten out.
- There are more combinations where a particle can do some lefts, then some rights and end up back where it started.
![One Dimensional Random Walk](images/random-walk-timestep-particle.png)
- What is the average displacement?:
![Average Displacement](images/displacement-calculate-equation.png)
- What this means is that the center of the distribution has not changed.
- Calculating the the average distance is simply the variance in Statistical terms:
![Particle Variance Calculation](images/particle-variance-calculation-equaton.png)
- This just means that the variance grows as a constant rate.
- **Diffusion Coefficient**:
![Diffusion Coefficient](images/derive-diffusion-coefficient.png)
- **The Distance that a particle diffuses - on average, from it's starting point - grows with the square root of time.**
- Now we'll look at **Fisk's First Law**:
  1. Diffusion produces a net flow of particles from regions of high concentration to regions of lower concentration.
  2. The flux of particles is proportional to the concentration gradient.
- If you divide by the distance then it looks like a derivative:
![Fick's First Law Derivation](images/ficks-first-law-derivation.png)
- Current flow in neuron's obeys Ohm's Law.
![Ohm's Law](images/neurons-obey-ohms-lay.png)
![Where Does Ohms Law Come From?](images/where-does-ohms-law-come-from.png)
![Eintstein-Smoluchovski Relation](images/einstein-smoluchovski-relation.png)
- Resistivity in the brain is really lousy.
- Due to this, the brain has to create this gimmick - an **Action Potential** to actually send anything.


# RC Circuit and Nernst Potential

# Nernst Potential and Integrate and Fire Models

# Hodgkin-Huxley Model Part 1

# Hodgkin-Huxley Model Part 2

# Dendrites

# Synapses

# Spike Trains

# Receptive Fields

# Time Series

# Spectral Analysis Part 1

# Spectral Analysis Part 2

# Spectral Analysis Part 3

# Rate Models and Perceptrons

# Matrix Operations

# Basis Sets

# Principal Components Analysis​

# Recurrent Networks

#  Neural Integrators

# Hopfield Networks

# Research:

# Reference:
