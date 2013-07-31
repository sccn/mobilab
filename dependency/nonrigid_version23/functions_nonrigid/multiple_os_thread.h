/*   undef needed for LCC compiler  */
#undef EXTERN_C
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define voidthread unsigned __stdcall
#define EndThread _endthreadex(0); return 0
#define ThreadHANDLE HANDLE
#define WaitForThreadFinish(t) WaitForSingleObject(t, INFINITE);CloseHandle(t)
#define StartThread(a,b,c) a=(HANDLE)_beginthreadex(NULL, 0, b, c, 0, NULL );
#else
#include <pthread.h>
#define voidthread void
#define EndThread pthread_exit(NULL)
#define ThreadHANDLE pthread_t
#define WaitForThreadFinish(t) pthread_join(t, NULL)
#define StartThread(a,b,c) pthread_create((pthread_t*)&a, NULL, (void*)b, c);
#endif
