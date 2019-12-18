/*
 * C. Kallidonis
 * This file follows the QUDA conventions for command-line options.
 * It serves as an extension in order to allow MUGIQ-related command-line options along with the QUDA ones.
 */

#include "test_params_mugiq.h"

MuGiqEigTask mugiq_eig_task = MUGIQ_COMPUTE_EVECS_INVALID;

namespace {
  CLI::TransformPairs<MuGiqEigTask> mugiq_eig_task_map {{"computeEvecsQuda", MUGIQ_COMPUTE_EVECS_QUDA},
							{"computeEvecs", MUGIQ_COMPUTE_EVECS_MUGIQ}};
}


// Options for Eigensolver within MuGiq
void add_eigen_option_mugiq(std::shared_ptr<QUDAApp> app)
{ 
  auto opgroup = app->add_option_group("Eigensolver-MuGiq", "Eigensolver Options within MuGiq");

  opgroup->add_option("--mugiq-eig-task", mugiq_eig_task,
		      "Task to perform in the eigensolve test, options are computeEvecs/computeEvecsQuda (default NULL)")->transform(CLI::QUDACheckedTransformer(mugiq_eig_task_map));
}



/*
std::shared_ptr<MUGIQApp> make_mugiq_app(std::string app_description, std::string app_name)
{
  auto mugiq_app = std::make_shared<MUGIQApp>(app_description, app_name);
  mugiq_app->option_defaults()->always_capture_default();

  return mugiq_app;
}
*/
