#include <inttypes.h>
#include <stdio.h>
#include <ctime>
int main(){
    srand(time(NULL));
	char *filename = "random.bin";
    long x;
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL){
		printf("Error opening the file %s", filename);
		return -1;
	}
	for(int i = 0; i < 1048576; i++){
        x = rand();
		fwrite (&x, sizeof (x), 1, fp);
	}
    fclose(fp);
    FILE * fh = fopen ("random.bin", "rb");
    for(int i = 0; i < 1048576; i++){
        if (fh != NULL) {
            fread (&x, sizeof (x), 1, fh);
            fclose (fh);
        }
        printf ("Value is: %d\n", x);
    }
	return 0;
}