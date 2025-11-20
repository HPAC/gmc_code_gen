#ifdef INSTRUMENT_CODE
extern int OPS_COUNTER;
#endif

#ifdef INSTRUMENT_CODE
#define COUNT(x) OPS_COUNTER = OPS_COUNTER + x;
#define RESET_COUNT OPS_COUNTER = 0;
#else
#define COUNT(x)
#define RESET_COUNT
#endif