/*
 * C. Kallidonis
 * This file follows the QUDA conventions for command-line options.
 * It serves as an extension in order to allow MUGIQ-related command-line options along with the QUDA ones.
 */

#include "test_params_mugiq.h"

MuGiqTask mugiq_task = MUGIQ_TASK_INVALID;
MuGiqEigOperator mugiq_eig_operator = MUGIQ_EIG_OPERATOR_INVALID;

namespace {
  CLI::TransformPairs<MuGiqTask> mugiq_task_map {{"computeEvecsQuda", MUGIQ_COMPUTE_EVECS_QUDA},
						 {"computeEvecs", MUGIQ_COMPUTE_EVECS_MUGIQ},
						 {"computeLoopULocal", MUGIQ_COMPUTE_LOOP_ULOCAL}};
  CLI::TransformPairs<MuGiqEigOperator> mugiq_eig_optr_map {{"mg", MUGIQ_EIG_OPERATOR_MG},
							    {"no_mg", MUGIQ_EIG_OPERATOR_NO_MG}};
}


// Options for Eigensolver within MuGiq
void add_eigen_option_mugiq(std::shared_ptr<QUDAApp> app)
{ 
  auto opgroup = app->add_option_group("Eigensolver-MuGiq", "Eigensolver Options within MuGiq");

  opgroup->add_option("--mugiq-task", mugiq_task,
		      "Task to perform in the eigensolve test, options are computeEvecs/computeEvecsQuda (default NULL)")->transform(CLI::QUDACheckedTransformer(mugiq_task_map));

  opgroup->add_option("--mugiq-eig-operator", mugiq_eig_operator,
		      "Operator of which to calculate eigen-pairs, options are no_mg/mg (default NULL)")->transform(CLI::QUDACheckedTransformer(mugiq_eig_optr_map));
}



/*
std::shared_ptr<MUGIQApp> make_mugiq_app(std::string app_description, std::string app_name)
{
  auto mugiq_app = std::make_shared<MUGIQApp>(app_description, app_name);
  mugiq_app->option_defaults()->always_capture_default();

  return mugiq_app;
}
*/
