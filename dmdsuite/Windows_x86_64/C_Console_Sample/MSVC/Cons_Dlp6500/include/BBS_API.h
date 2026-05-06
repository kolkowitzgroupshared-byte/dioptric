#pragma once


#define hDEV void*

//Error Codes
#define RESULT_OK 0
#define RESULT_ERROR_BAD_PARAM -1
#define RESULT_ERROR_TRANSMISSION -2
#define RESULT_ERROR_TIMEOUT	-3


//Other defines
#define BYTE unsigned char
#define uint unsigned int


#if ( defined DLLUSE )
#  define DLL_API __declspec( dllexport ) 
#else
#  define DLL_API __declspec( dllimport ) 
#endif


#define REG0 0
#define REG1 1

#include "API_defines.h"


EXTERN_C
{
	/*** IMAGE HANDLING ***/
	
	DLL_API	void CalcPlane(BYTE* pixelbuffer, BYTE* plainbuffer, int bitlevel);
	DLL_API void SplitRGBPixels(uint* rgbValues, BYTE* rValues, BYTE* gValues, BYTE* bValues);


	//functions that do not access a projector and need no device handle


	//! Number of the connected devices that have the FT600 USB chip and the product string DMD6500
	DLL_API int ListControllers(uint * count);

	//! List of IDs of the connected devices that have the FT600 USB chip and the product string DMD6500
	DLL_API int GetDevID(uint index, int * ID );


	//! Serial number of the device selected from the List received by GetNumDevs  and  GetDevID
	//! It is not needed to Connect before calling this function
	//! GetNumDevs must be called before calling this function
	DLL_API int GetSerialNumber(int ID, char* serialnumber);



	//Get a device handle.  if multiple projectors are used every projector needs its own device handle
	DLL_API hDEV GetDevice();

	DLL_API void DeleteDevice(hDEV device);




	//functions below  access a projector specified by a device handle


	/*** USB CONNECTION ***/


	DLL_API int Connect(hDEV devHandle, int devID);	

	DLL_API int Disconnect(hDEV devHandle);

	DLL_API int IsConnected(hDEV devHandle, bool* connected);
	
	
	DLL_API int GetFirmwareVersion(hDEV devHandle, uint  *version);



	/*** IMAGE TRANSFER INTO CONTROLLER MEMORY ***/

 																
	DLL_API int SendImageRGB(hDEV devHandle, int planenr, uint* ImgBuffer);	

	DLL_API int SendImageMono(hDEV devHandle, int planenr, BYTE* PixBuffer);
	
	DLL_API int SendPlane(hDEV devHandle, int planenr, BYTE* PlaneBuffer);
	
	
	///*** DMD CONTROL ***/
	
	DLL_API int LoadPlaneToDLP(hDEV devHandle, int planenr);
	
	DLL_API int DLP_GlobalMCP(hDEV devHandle);

	DLL_API int WriteCommand(hDEV devHandle, uint seq_cmd);
	
	
	///*** SETUP ***/
	
	DLL_API int SetModeRegister(hDEV devHandle, int regnr, int value);
	
	DLL_API int StoreLogoToFlash(hDEV devHandle, int planenr);

	DLL_API int GetMaxPlaneNr(hDEV devHandle, uint  *maxplanenr);



	///*** SEQUENCE ***/										

	DLL_API int RunSequence(hDEV devHandle, int startpos);		

	DLL_API int StopSequence(hDEV devHandle);

	DLL_API int SendSequenceData(hDEV Proj,  uint* buffer, int bufoffset, int cmdcnt, int startpos);
																						
	DLL_API int IsSequenceRunning(hDEV devHandle, bool *value);														

	
}


