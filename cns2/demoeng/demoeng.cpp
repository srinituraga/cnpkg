#include <stdio.h>
#include <stdarg.h>
#include "engine.h"

#define BUFSIZE 4096

bool RunDemo(Engine *ep, char *out);

bool Eval(Engine *ep, char *out, const char *cmd, ...);

/**********************************************************************************************************************/

int main(int argc, char *argv[]) {

    printf("\n");

    if (argc < 2) {
        printf("not enough arguments\n\n");
        return 1;
    }

    printf("STARTING MATLAB ENGINE\n\n");
    Engine *ep = engOpen("");
    if (ep == NULL) {
        printf("unable to start MATLAB engine\n\n");
        return 1;
    }
    engSetVisible(ep, false);
    char out[BUFSIZE];
    engOutputBuffer(ep, out, BUFSIZE);

    printf("SETTING PATH TO CNS\n\n");
    Eval(ep, out, "run(fullfile('%s', 'cns_path'));", argv[1]);

    bool ok = RunDemo(ep, out);

    printf("PRESS RETURN TO CONTINUE: ");
    fgets(out, BUFSIZE, stdin);
    printf("\n");

    printf("CLOSING MATLAB ENGINE\n\n");
    Eval(ep, out, "close all;");
    engClose(ep);

    return ok ? 0 : 1;

}

/**********************************************************************************************************************/

bool RunDemo(Engine *ep, char *out) {

    printf("RUNNING DEMO SCRIPT\n\n");
    if (!Eval(ep, out, "demopkg_run;")) return false;
    printf("\n");

    printf("RETRIEVING SOME OUTPUT\n\n");
    mxArray *res = engGetVariable(ep, "res");
    if (res == NULL) {
        printf("unable to retrieve variable 'res'\n\n");
        return false;
    }
    for (int i = 0; i < 10; i++) {
        float f = ((float *)mxGetData(res))[i];
        printf("res[%i] = %f\n", i, f);
    }
    printf("\n");
    mxDestroyArray(res);

    return true;

}

/**********************************************************************************************************************/

bool Eval(Engine *ep, char *out, const char *cmd, ...) {

    char buf[BUFSIZE];
    va_list args;
    va_start(args, cmd);
    vsprintf(buf, cmd, args);
    va_end(args);

    engEvalString(ep, "clear engerr; lasterr('');");

    engEvalString(ep, buf);
    printf("%s", out);

    engEvalString(ep, "engerr = lasterr;");

    mxArray *err = engGetVariable(ep, "engerr");
    if (err == NULL) {
        printf("unable to retrieve variable 'engerr'\n\n");
        return false;
    }
    bool ok = (mxGetNumberOfElements(err) == 0);
    mxDestroyArray(err);

    return ok;

}
