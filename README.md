# Quasi-Monte Carlo Community Software

<hr>

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: an integrand, a discrete distribution, summary output data, and stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications. 

<hr>

## Citation

If you find QMCPy helpful in your work, please support us by citing the
following work:

Fred J. Hickernell, Sou-Cheng T. Choi, and Aleksei Sorokin, 
“QMC  Community Software.” Python software, 2019. Work in progress. 
Available on [GitHub](https://github.com/QMCSoftware/QMCSoftware)

<hr>

## Developers
 
- Sou-Cheng T. Choi
- Fred J. Hickernell
- Aleksei Sorokin

<hr>

## Contributors

- Michael McCourt

<hr>

## Acknowledgment 

We thank Dirk Nuyens for fruitful discussions related to Magic Point Shop.


<hr>

## References

1. F.Y. Kuo & D. Nuyens. Application of quasi-Monte Carlo methods to elliptic 
PDEs with random diffusion coefficients - a survey of analysis and 
implementation, Foundations of Computational Mathematics, 16(6):1631-1696, 2016. 
([springer link](https://link.springer.com/article/10.1007/s10208-016-9329-5), 
[arxiv link](https://arxiv.org/abs/1606.06613))