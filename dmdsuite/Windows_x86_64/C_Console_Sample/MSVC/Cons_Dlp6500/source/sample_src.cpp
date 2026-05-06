
#include "stdafx.h"
#include "windows.h"
#include <time.h>


#include "Gdiplus.h"

#include "BBS_API.h"

#include "stdio.h"

using namespace std;



BYTE buffer[64];


#include <malloc.h>





int main()
{

	uint ndevices;
	uint SeqBufferPtr[SEQ_BUFFER_SIZE];

	int k=0;

	int seq[10];

	int sn = 0;


	if(1)//optional check if device is present
	{
		ListControllers( & ndevices );	printf("Devices available = %d\n", ndevices);

		if(ndevices == 0)
		{
			printf("No Devices found exiting!\n");
			getchar();
			return 0;
		}		//no devices found	
	}




	//opend the device
	hDEV devhandle =  GetDevice();

	Connect( devhandle, 0 );


	
	//create a 8 bit monochrome ramp testimage in memory
	BYTE* imgbuffer = new BYTE[1920*1080];

	for(int y=0; y<1080; y++)
	{
		for(int x=0; x<1920; x++)
		{
			imgbuffer[x+1920*y] =  x % 256;
		}
	}

	//send the image to the controller into plane nr 0..7

	SendImageMono(devhandle, 0, imgbuffer);

	delete imgbuffer;


	
	StopSequence( devhandle);	//to be sure


	//show the ramp image using the firmware defined grayscale sequence
	RunSequence( devhandle,  0 );

	
	printf("Press Enter to continue\n");
	getchar();


	//define a sequence to show the 8 bitplanes slowly
	if(1)
	{
		seq[sn++] = k;
		
		int kstart = k;

		for(int i=0; i< 8 ; i++)
		{

			SeqBufferPtr[k++] = CMD_GLOB_LOAD | LOAD_PLANE_NR | i;

			SeqBufferPtr[k++] = CMD_GLOB_MIRRORCLOCKING;

			SeqBufferPtr[k++] = CMD_WAIT_US_SINCE_MCP | 200000;
		}

		SeqBufferPtr[k++] = (CMD_SEQ_END);	
	}


	StopSequence( devhandle);

	//Copy the Sequence into the controller SequenceBuffer into position 1000
	SendSequenceData(devhandle,  SeqBufferPtr, 0 , k, 1000);
	
	//Run the Sequence at position 1000 to show the planes
	RunSequence( devhandle,  1000 );


		
	printf("Press Enter to continue\n");
	getchar();

	StopSequence( devhandle);

	RunSequence( devhandle,  IDLE_SEQUENCE_START );


	//cleanup before closing
	Disconnect( devhandle );

	DeleteDevice( devhandle );

}