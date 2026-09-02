1. activation functions - softmax? global inhibition? - softmax best for grid layer, tanh for sensory, but softmax is biologically unrealistic, so glob inh is it, even though scaffold fails with it - we try to see if adding theta with phases can recover memory functions.
2. which numerical integration method - RK4
3. how to decide taus - literature? - 10ms, based on Ji et al., 2025 - but timescales only important when including theta waves
5. make a separate file for helper functions - done

still todo
4. add theta (square wave) terms to scaffold with glob inh - phases?

make jax code:
1. where should the activation function be defined? within the compiled code?
2. verify the rk4 steps
3. try one run
4. try fig 2e (pcorrect) for scaffold
5. VERY IMP: memory continuum plot

