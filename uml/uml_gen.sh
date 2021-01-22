#cd uml
#./uml_gen.sh

pyreverse -k ../qmcpy/ --ignore util,stopping_criterion,accumulate_data,discrete_distribution; mv classes.dot qmcpy1.dot; dot -Tpng qmcpy1.dot > qmcpy_uml1.png
pyreverse -k ../qmcpy/discrete_distribution/; mv classes.dot qmcpy2.dot; dot -Tpng qmcpy2.dot > qmcpy_uml2.png
pyreverse -k ../qmcpy/ --ignore util,integrand,true_measure,discrete_distribution; mv classes.dot qmcpy3.dot; dot -Tpng qmcpy3.dot > qmcpy_uml3.png

pyreverse ../qmcpy/accumulate_data/; mv classes.dot accumulate_data.dot; dot -Tpng accumulate_data.dot > accumulate_data_uml.png

pyreverse ../qmcpy/discrete_distribution/; mv classes.dot discrete_distribution.dot; dot -Tpng discrete_distribution.dot > discrete_distribution_uml.png

pyreverse ../qmcpy/true_measure/; mv classes.dot true_measure.dot; dot -Tpng true_measure.dot > true_measure_uml.png

pyreverse ../qmcpy/integrand/; mv classes.dot integrand.dot; dot -Tpng integrand.dot > integrand_uml.png

pyreverse ../qmcpy/stopping_criterion/; mv classes.dot stopping_criterion.dot; dot -Tpng stopping_criterion.dot > stopping_criterion_uml.png

pyreverse -k ../qmcpy/util/; mv classes.dot util.dot; dot -Tpng util.dot > util_uml.png
# manually modify util.dot to produce util_warn.dot and util_err.dot
dot -Tpng util_warn.dot > util_warn.png
dot -Tpng util_err.dot > util_err.png

#cd ..
#make doc_html
#make doc_pdf
