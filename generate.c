#include <inttypes.h>
#include <stdio.h>
int main(){
	char *filename = "random.txt";
    long x;
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL){
		printf("Error opening the file %s", filename);
		return -1;
	}
	for(int i = 0; i < 1048576; i++){
        x = (long) rand();
		fwrite (&x, sizeof (x), 1, fp);
	}
	fclose(fp);
	return 0;
}