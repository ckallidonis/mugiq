/*
 * C. Kallidonis
 * This file follows the QUDA conventions for command-line options.
 * It serves as an extension in order to allow MUGIQ-related command-line options along with the QUDA ones.
 */

#include <array>
#include <externals/CLI11.hpp> // This is a QUDA include file
#include <test_params.h>  // This is a QUDA include file
#include <mugiq.h>

void add_eigen_option_mugiq(std::shared_ptr<QUDAApp> app);
void add_loop_option_mugiq(std::shared_ptr<QUDAApp> app);


//- External variables used in tests
extern MuGiqTask mugiq_task;
extern MuGiqBool mugiq_use_mg;

extern char mugiq_mom_filename[1024];
extern LoopFTSign loop_ft_sign;
extern LoopCalcType loop_calc_type;
extern MuGiqBool loop_write_mom_space_hdf5;
extern MuGiqBool loop_write_pos_space_hdf5;
extern MuGiqBool loop_doMomProj;
extern MuGiqBool loop_doNonLocal;
extern MuGiqBool compute_coarse;
extern char loop_gauge_filename[1024];
extern std::string disp_entry_string;
extern std::string fname_mom_h5;
extern std::string fname_pos_h5;



/*
class MUGIQApp : public CLI::App
{
public:
  MUGIQApp(std::string app_description = "", std::string app_name = "") : CLI::App(app_description, app_name) {};
  
  virtual ~MUGIQApp() {};
};

std::shared_ptr<MUGIQApp> make_mugiq_app(std::string app_description = "MUGIQ test", std::string app_name = "");
void add_eigen_option_mugiq(std::shared_ptr<MUGIQApp> mugiq_app);
*/

