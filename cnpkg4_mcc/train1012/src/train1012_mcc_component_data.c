/*
 * MATLAB Compiler: 4.11 (R2009b)
 * Date: Mon Aug 22 16:59:16 2011
 * Arguments: "-B" "macro_default" "-o" "train1012" "-W" "main" "-d"
 * "/net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4_mcc/train1012/src" "-T"
 * "link:exe" "-v"
 * "/net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4_mcc/train1012.m" "-a"
 * "/net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4" "-a"
 * "/net/omicfs/home/sturaga/gpu/jimGPUnet/cns2" 
 */

#include "mclmcrrt.h"

#ifdef __cplusplus
extern "C" {
#endif
const unsigned char __MCC_train1012_session_key[] = {
    '5', '9', 'B', 'E', 'F', 'C', '6', '9', '1', '3', 'B', '0', '1', '6', '8',
    'F', '1', '3', 'E', '3', '2', '9', 'C', 'E', '5', '6', '6', '4', '0', 'A',
    '3', '2', '2', 'F', '3', 'E', '0', 'B', 'B', 'C', '0', 'F', '0', 'B', 'C',
    'D', '9', '8', '2', '9', 'A', 'F', '7', 'D', 'D', 'F', '7', 'F', 'F', 'C',
    '8', '8', '9', '5', 'D', '1', '8', '7', 'D', 'F', '2', '1', 'B', 'B', '2',
    '4', '6', '7', '2', 'D', '7', '1', '8', 'D', 'A', '3', '7', '1', '8', '3',
    '3', 'B', 'A', '8', '4', '1', '6', '4', '3', '3', '4', '3', '0', '7', '4',
    '3', '3', 'F', '8', 'D', '5', '3', '7', 'B', '8', 'B', '6', 'D', '5', '8',
    '4', '1', '3', '7', 'C', 'C', '9', '2', '3', 'D', 'E', 'E', '3', '5', '2',
    '8', '8', 'D', '1', 'D', '4', '2', 'F', '2', '8', 'C', 'A', '9', '5', 'C',
    'D', 'B', 'A', '4', '2', 'F', '8', '7', 'F', '8', 'F', '3', '7', '6', '9',
    '7', '3', 'E', '8', 'B', '8', '5', '2', '1', 'C', 'C', 'B', 'C', '0', '3',
    '6', '1', 'D', '6', '4', '4', 'D', '7', '2', '3', 'C', '1', 'B', '7', 'D',
    'D', '5', '7', 'F', 'B', '3', '5', '9', 'B', '5', '2', '7', 'E', '2', '9',
    '3', '2', '1', 'D', '3', 'E', '3', '0', '1', '1', '2', 'E', 'C', 'B', '4',
    '3', '3', 'E', '9', '9', '5', '1', '6', 'F', '7', '0', '5', '0', 'A', 'C',
    'E', 'B', '2', '1', 'F', 'D', '1', 'D', '4', 'A', 'C', '0', '6', 'A', '6',
    'D', '\0'};

const unsigned char __MCC_train1012_public_key[] = {
    '3', '0', '8', '1', '9', 'D', '3', '0', '0', 'D', '0', '6', '0', '9', '2',
    'A', '8', '6', '4', '8', '8', '6', 'F', '7', '0', 'D', '0', '1', '0', '1',
    '0', '1', '0', '5', '0', '0', '0', '3', '8', '1', '8', 'B', '0', '0', '3',
    '0', '8', '1', '8', '7', '0', '2', '8', '1', '8', '1', '0', '0', 'C', '4',
    '9', 'C', 'A', 'C', '3', '4', 'E', 'D', '1', '3', 'A', '5', '2', '0', '6',
    '5', '8', 'F', '6', 'F', '8', 'E', '0', '1', '3', '8', 'C', '4', '3', '1',
    '5', 'B', '4', '3', '1', '5', '2', '7', '7', 'E', 'D', '3', 'F', '7', 'D',
    'A', 'E', '5', '3', '0', '9', '9', 'D', 'B', '0', '8', 'E', 'E', '5', '8',
    '9', 'F', '8', '0', '4', 'D', '4', 'B', '9', '8', '1', '3', '2', '6', 'A',
    '5', '2', 'C', 'C', 'E', '4', '3', '8', '2', 'E', '9', 'F', '2', 'B', '4',
    'D', '0', '8', '5', 'E', 'B', '9', '5', '0', 'C', '7', 'A', 'B', '1', '2',
    'E', 'D', 'E', '2', 'D', '4', '1', '2', '9', '7', '8', '2', '0', 'E', '6',
    '3', '7', '7', 'A', '5', 'F', 'E', 'B', '5', '6', '8', '9', 'D', '4', 'E',
    '6', '0', '3', '2', 'F', '6', '0', 'C', '4', '3', '0', '7', '4', 'A', '0',
    '4', 'C', '2', '6', 'A', 'B', '7', '2', 'F', '5', '4', 'B', '5', '1', 'B',
    'B', '4', '6', '0', '5', '7', '8', '7', '8', '5', 'B', '1', '9', '9', '0',
    '1', '4', '3', '1', '4', 'A', '6', '5', 'F', '0', '9', '0', 'B', '6', '1',
    'F', 'C', '2', '0', '1', '6', '9', '4', '5', '3', 'B', '5', '8', 'F', 'C',
    '8', 'B', 'A', '4', '3', 'E', '6', '7', '7', '6', 'E', 'B', '7', 'E', 'C',
    'D', '3', '1', '7', '8', 'B', '5', '6', 'A', 'B', '0', 'F', 'A', '0', '6',
    'D', 'D', '6', '4', '9', '6', '7', 'C', 'B', '1', '4', '9', 'E', '5', '0',
    '2', '0', '1', '1', '1', '\0'};

static const char * MCC_train1012_matlabpath_data[] = 
  { "train1012/", "$TOOLBOXDEPLOYDIR/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/util/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/util/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/util/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/.svn/tmp/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/.settings/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/.settings/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/.settings/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/bin/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/bin/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/bin/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/bin/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/src/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/src/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/src/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/demo/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/demo/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/demo/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/demo/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demopkg/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/linux/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/linux/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/linux/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/linux/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/win/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/win/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/win/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac_old/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac_old/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac_old/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/mac_old/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/scripts/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/figs/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/figs/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/figs/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/figs/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/doc/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demoeng/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demoeng/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demoeng/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/demoeng/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/deprecated/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/deprecated/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/deprecated/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/source/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/source/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/source/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/private/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/private/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/private/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/util/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/util/private/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/util/private/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/util/private/.svn/prop-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/util/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/util/.svn/text-base/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/.svn/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cns2/.svn/text-base/",
    "net/omicfs/home/sturaga/EM/util/", "net/omicfs/home/sturaga/EM/",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg/",
    "net/omicfs/home/sturaga/matlab/", "$TOOLBOXMATLABDIR/general/",
    "$TOOLBOXMATLABDIR/ops/", "$TOOLBOXMATLABDIR/lang/",
    "$TOOLBOXMATLABDIR/elmat/", "$TOOLBOXMATLABDIR/randfun/",
    "$TOOLBOXMATLABDIR/elfun/", "$TOOLBOXMATLABDIR/specfun/",
    "$TOOLBOXMATLABDIR/matfun/", "$TOOLBOXMATLABDIR/datafun/",
    "$TOOLBOXMATLABDIR/polyfun/", "$TOOLBOXMATLABDIR/funfun/",
    "$TOOLBOXMATLABDIR/sparfun/", "$TOOLBOXMATLABDIR/scribe/",
    "$TOOLBOXMATLABDIR/graph2d/", "$TOOLBOXMATLABDIR/graph3d/",
    "$TOOLBOXMATLABDIR/specgraph/", "$TOOLBOXMATLABDIR/graphics/",
    "$TOOLBOXMATLABDIR/uitools/", "$TOOLBOXMATLABDIR/strfun/",
    "$TOOLBOXMATLABDIR/imagesci/", "$TOOLBOXMATLABDIR/iofun/",
    "$TOOLBOXMATLABDIR/audiovideo/", "$TOOLBOXMATLABDIR/timefun/",
    "$TOOLBOXMATLABDIR/datatypes/", "$TOOLBOXMATLABDIR/verctrl/",
    "$TOOLBOXMATLABDIR/codetools/", "$TOOLBOXMATLABDIR/helptools/",
    "$TOOLBOXMATLABDIR/demos/", "$TOOLBOXMATLABDIR/timeseries/",
    "$TOOLBOXMATLABDIR/hds/", "$TOOLBOXMATLABDIR/guide/",
    "$TOOLBOXMATLABDIR/plottools/", "toolbox/local/",
    "toolbox/shared/dastudio/", "$TOOLBOXMATLABDIR/datamanager/",
    "toolbox/compiler/", "toolbox/images/colorspaces/",
    "toolbox/images/images/", "toolbox/images/imuitools/",
    "toolbox/images/iptformats/", "toolbox/images/iptutils/",
    "toolbox/shared/imageslib/" };

static const char * MCC_train1012_classpath_data[] = 
  { "java/jar/toolbox/images.jar",
    "net/omicfs/home/sturaga/gpu/jimGPUnet/cnpkg4/DAGsurgery/bin/DAGedit.jar" };

static const char * MCC_train1012_libpath_data[] = 
  { "bin/glnxa64/" };

static const char * MCC_train1012_app_opts_data[] = 
  { "" };

static const char * MCC_train1012_run_opts_data[] = 
  { "" };

static const char * MCC_train1012_warning_state_data[] = 
  { "off:MATLAB:dispatcher:nameConflict" };


mclComponentData __MCC_train1012_component_data = { 

  /* Public key data */
  __MCC_train1012_public_key,

  /* Component name */
  "train1012",

  /* Component Root */
  "",

  /* Application key data */
  __MCC_train1012_session_key,

  /* Component's MATLAB Path */
  MCC_train1012_matlabpath_data,

  /* Number of directories in the MATLAB Path */
  121,

  /* Component's Java class path */
  MCC_train1012_classpath_data,
  /* Number of directories in the Java class path */
  2,

  /* Component's load library path (for extra shared libraries) */
  MCC_train1012_libpath_data,
  /* Number of directories in the load library path */
  1,

  /* MCR instance-specific runtime options */
  MCC_train1012_app_opts_data,
  /* Number of MCR instance-specific runtime options */
  0,

  /* MCR global runtime options */
  MCC_train1012_run_opts_data,
  /* Number of MCR global runtime options */
  0,
  
  /* Component preferences directory */
  "train1012_A1E3CAB26F44A1C83B3F7EB843449676",

  /* MCR warning status data */
  MCC_train1012_warning_state_data,
  /* Number of MCR warning status modifiers */
  1,

  /* Path to component - evaluated at runtime */
  NULL

};

#ifdef __cplusplus
}
#endif


