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
#include <stdio.h>
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" {
#endif

extern mclComponentData __MCC_train1012_component_data;

#ifdef __cplusplus
}
#endif

static HMCRINSTANCE _mcr_inst = NULL;

#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultPrintHandler(const char *s)
{
  return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultErrorHandler(const char *s)
{
  int written = 0;
  size_t len = 0;
  len = strlen(s);
  written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
  if (len > 0 && s[ len-1 ] != '\n')
    written += mclWrite(2 /* stderr */, "\n", sizeof(char));
  return written;
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

#ifndef LIB_train1012_C_API
#define LIB_train1012_C_API /* No special import/export declaration */
#endif

LIB_train1012_C_API 
bool MW_CALL_CONV train1012InitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
  if (_mcr_inst != NULL)
    return true;
  if (!mclmcrInitialize())
    return false;
  if (!mclInitializeComponentInstanceWithEmbeddedCTF(&_mcr_inst, 
                                                     &__MCC_train1012_component_data, 
                                                     true, NoObjectType, ExeTarget, 
                                                     error_handler, print_handler, 
                                                     3176152, (void 
                                                     *)(train1012InitializeWithHandlers)))
    return false;
  return true;
}

LIB_train1012_C_API 
bool MW_CALL_CONV train1012Initialize(void)
{
  return train1012InitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}
LIB_train1012_C_API 
void MW_CALL_CONV train1012Terminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

int run_main(int argc, const char **argv)
{
  int _retval;
  /* Generate and populate the path_to_component. */
  char path_to_component[(PATH_MAX*2)+1];
  separatePathName(argv[0], path_to_component, (PATH_MAX*2)+1);
  __MCC_train1012_component_data.path_to_component = path_to_component; 
  if (!train1012Initialize()) {
    return -1;
  }
  argc = mclSetCmdLineUserData(mclGetID(_mcr_inst), argc, argv);
  _retval = mclMain(_mcr_inst, argc, argv, "train1012", 0);
  if (_retval == 0 /* no error */) mclWaitForFiguresToDie(NULL);
  train1012Terminate();
  mclTerminateApplication();
  return _retval;
}

int main(int argc, const char **argv)
{
  if (!mclInitializeApplication(
    __MCC_train1012_component_data.runtime_options, 
    __MCC_train1012_component_data.runtime_option_count))
    return 0;

  return mclRunMain(run_main, argc, argv);
}
