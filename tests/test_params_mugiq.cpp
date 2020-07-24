/*
 * C. Kallidonis
 * This file follows the QUDA conventions for command-line options.
 * It serves as an extension in order to allow MUGIQ-related command-line options along with the QUDA ones.
 */

#include "test_params_mugiq.h"

MuGiqTask mugiq_task = MUGIQ_TASK_INVALID;
MuGiqBool mugiq_use_mg = MUGIQ_BOOL_INVALID;

char mugiq_mom_filename[1024] = "momenta.txt";
LoopFTSign loop_ft_sign = LOOP_FT_SIGN_INVALID;
LoopCalcType loop_calc_type = LOOP_CALC_TYPE_INVALID;
MuGiqBool loop_print_ascii = MUGIQ_BOOL_FALSE;
MuGiqBool loop_doMomProj = MUGIQ_BOOL_TRUE;
MuGiqBool loop_doNonLocal = MUGIQ_BOOL_TRUE;
char loop_gauge_filename[1024] = "";
char loop_path_string[1024] = "";

namespace {
  CLI::TransformPairs<MuGiqTask> mugiq_task_map {{"computeEvecsQuda", MUGIQ_COMPUTE_EVECS_QUDA},
						 {"computeEvecs", MUGIQ_COMPUTE_EVECS_MUGIQ},
						 {"computeLoopULocal", MUGIQ_COMPUTE_LOOP_ULOCAL}};
  
  CLI::TransformPairs<MuGiqBool> mugiq_use_mg_map {{"yes", MUGIQ_BOOL_TRUE},
						   {"no", MUGIQ_BOOL_FALSE}};

  CLI::TransformPairs<LoopFTSign> loop_ft_sign_map {{"plus", LOOP_FT_SIGN_PLUS},
						    {"minus", LOOP_FT_SIGN_MINUS}};

  CLI::TransformPairs<LoopCalcType> loop_calc_type_map {{"blas",  LOOP_CALC_TYPE_BLAS},
							{"opt" ,  LOOP_CALC_TYPE_OPT_KERNEL},
							{"basic", LOOP_CALC_TYPE_BASIC_KERNEL}};

  CLI::TransformPairs<MuGiqBool> loop_print_ascii_map {{"yes",  MUGIQ_BOOL_TRUE},
						       {"no" ,  MUGIQ_BOOL_FALSE}};
  
  CLI::TransformPairs<MuGiqBool> loop_doMomProj_map {{"yes",  MUGIQ_BOOL_TRUE},
						     {"no" ,  MUGIQ_BOOL_FALSE}};

  CLI::TransformPairs<MuGiqBool> loop_doNonLocal_map {{"yes",  MUGIQ_BOOL_TRUE},
						      {"no" ,  MUGIQ_BOOL_FALSE}};
  
}


// Options for Eigensolver within MuGiq
void add_eigen_option_mugiq(std::shared_ptr<QUDAApp> app)
{ 
  auto opgroup = app->add_option_group("Eigensolver-MuGiq", "Eigensolver Options within MuGiq");

  opgroup->add_option("--mugiq-task", mugiq_task,
		      "Task to perform in the eigensolve test, options are computeEvecs/computeEvecsQuda (default NULL)")->transform(CLI::QUDACheckedTransformer(mugiq_task_map));
  
  opgroup->add_option("--mugiq-use-mg", mugiq_use_mg,
		      "Whether to use MG in Eigenpair calculation, options are yes/no (default NULL)")->transform(CLI::QUDACheckedTransformer(mugiq_use_mg_map));
}


// Options for Loop Calculation
void add_loop_option_mugiq(std::shared_ptr<QUDAApp> app)
{ 
  auto opgroup = app->add_option_group("Loop-MuGiq", "Loop Options within MuGiq");
  
  opgroup->add_option("--momenta-filename", mugiq_mom_filename, "Filename with the momenta for Fourier Transform of the loop (default 'momenta.txt')");

  opgroup->add_option("--loop-gauge-filename", loop_gauge_filename, "Gauge field that will be used for non-local currents (default '')");

  opgroup->add_option("--loop-path-string", loop_path_string, "String with non-local current paths (default '')");

  opgroup->add_option("--loop-ft-sign", loop_ft_sign,
		      "Sign of the Loop Fourier Transform phase (default NULL)")->transform(CLI::QUDACheckedTransformer(loop_ft_sign_map));
  
  opgroup->add_option("--loop-calc-type", loop_calc_type,
		      "Type of loop calculation (default NULL, options are blas/opt/basic)")->transform(CLI::QUDACheckedTransformer(loop_calc_type_map));
  
  opgroup->add_option("--loop-print-ascii", loop_print_ascii,
		      "Whether to write loop in ASCII files (default no, options are yes/no)")->transform(CLI::QUDACheckedTransformer(loop_print_ascii_map));

  opgroup->add_option("--loop-do-momproj", loop_doMomProj,
		      "Whether to perform momentum projection (Fourier Transform) on the disconnected quark loop (default yes, options are yes/no)")->transform(CLI::QUDACheckedTransformer(loop_doMomProj_map));

  opgroup->add_option("--loop-do-nonlocal", loop_doNonLocal,
		      "Whether to compute quark loops for non-local currents, requires option --loop-gauge-filename and --loop-path-string (default yes, options are yes/no)")->transform(CLI::QUDACheckedTransformer(loop_doNonLocal_map));  
  
}



/*
std::shared_ptr<MUGIQApp> make_mugiq_app(std::string app_description, std::string app_name)
{
  auto mugiq_app = std::make_shared<MUGIQApp>(app_description, app_name);
  mugiq_app->option_defaults()->always_capture_default();

  return mugiq_app;
}
*/
