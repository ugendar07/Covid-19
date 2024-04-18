# COVID-19 Spread Prediction Model

## Overview
This repository contains a mathematical model developed to predict the spread of COVID-19 based on the rates of change of the number of susceptible, exposed, infected, and recovered individuals in a population. The model is parameterized using data from the COVID-19 pandemic in Karnataka, India. Additionally, it investigates the effects of different control measures, such as social distancing and vaccination, on the spread of the disease.


## Model Description
The model is a compartmental model, dividing the population into different compartments based on their infection status: susceptible (S), exposed (E), infected (I), and recovered (R). The model equations describe the flow of individuals between these compartments over time, taking into account various factors such as transmission rates, incubation period, and recovery rates.


 ### Data Set
- COVID-19 [data](https://courses.iisc.ac.in/pluginfile.php/71590/mod_assign/introattachment/0/COVID19_data.csv?forcedownload=1)
 
### Parameter Update Equations

The following equations describe the parameter update process using gradient descent:

```plaintext
β = β - 0.00001 ∂Loss/∂β

CIR0 = CIR0 - 0.00001 ∂Loss/∂CIR0

R0 = R0 - 0.00001 ∂Loss/∂R0

I0 = I0 - 0.00001 ∂Loss/∂I0

E0 = E0 - 0.00001 ∂Loss/∂E0

```

### Differential Equations Describing the Dynamics of COVID-19 Spread

The spread of COVID-19 in a population can be modeled using the following system of differential equations:

```plaintext
ΔS(t) = -β(t) * S(t) * I(t) / N - ε * ΔV(t) + ΔW(t)

ΔE(t) = β(t) * S(t) * I(t) / N - α * E(t)

ΔI(t) = α * E(t) - γ * I(t)

ΔR(t) = γ * I(t) + ε * ΔV(t) - ΔW(t)

```
## Results

The optimal unknown parameters are as follows:
- The optimal β value is = 0.456
- The optimal R_0 value is = 32%
- The optimal CIR_0 value is = 14.17
- The optimal I_0 value is = 0.34%
- The optimal C_0 value is = 0.49%


## Contact
For questions or inquiries, please contact [ugendar](mailto:ugendar07@gmail.com) .
