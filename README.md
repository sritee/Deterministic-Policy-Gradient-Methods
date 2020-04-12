# Deterministic Policy Gradient
This is a C++ implementation of a Deterministic Policy Gradient algorithm proposed by Silver et al [1]. We use tile coding proposed by Richard Sutton for the critic's linear function approximator. Note that this algorithm is different from Deep Deterministic Policy Gradient, as we use linear function approximation, and hence there are convergence guarantees. We test our algorithm on the Continuous Action Mountain Car domain, implemented similar to the OpenAI gym environment. 

For a detailed discussion, please visit my blog post [2]. 

<p align="center">
  <img src="https://github.com/sritee/Deterministic-Policy-Gradient-Methods/blob/master/plots.png" width="450" title="Continuous Action Mountain Car">

<p align="center"> 
   <b>References</b>
   </p>
   
 * [1] David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. "Deterministic policy gradient algorithms." In ICML. 2014.
  * [2] https://sridhartee.blogspot.in/2017/02/deterministic-policy-gradient-methods.html

