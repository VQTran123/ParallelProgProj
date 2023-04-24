#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(){
    time_t t;
    srand((unsigned) time(&t));
	char *filename = "random.bin";
    long x;
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL){
		printf("Error opening the file %s", filename);
		return -1;
	}
	for(int i = 0; i < 1048576; i++){
        x = rand() % 1000000;
        printf ("Value is: %ld\n", x);
		fwrite (&x, sizeof (x), 1, fp);
	}
    fclose(fp);
	return 0;
}
