#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>



void cartesian_to_barycentric(double *bary_coords, double P[2], double A[2], double B[2], double C[2])
{    
    // Define A as the origin and shift other points accordingly
    double v0[2], v1[2], v2[2];
    for(int i=0; i< 2; i+=1){
        v0[i] = B[i]-A[i];
        v1[i] = C[i]-A[i];
        v2[i] = P[i]-A[i];
    }
    
    // Calculate bnarycentric coords
    double normalizer = 1/(v0[0]*v1[1]-v1[0]*v0[1]);
    bary_coords[1] = (v2[0]*v1[1]-v1[0]*v2[1])*normalizer;
    bary_coords[2] = (v0[0]*v2[1]-v2[0]*v0[1])*normalizer;
    bary_coords[0] = 1 - bary_coords[1] - bary_coords[2];
}

bool in_triangle(double bary_coords[3])
{
    bool cond = true;
    for (int i=0; i<3; i+=1) {
        if (bary_coords[i] < 0) { cond=false; }
        if (bary_coords[i] > 1) { cond=false; }
    }
    return cond;
}

double * rasterize_trimesh(double * raster, const double * features, const double * xyz, const int * triangles, 
                           int CROPSIZE, int num_points, int num_features, int num_triangles)

{
    // Declare array that will store rasterized image values (2 channels for depth and ir)
    for(int i=0; i< CROPSIZE*CROPSIZE*num_features; i+=1){ raster[i]=0; }
    
    // Loop through triangles and add to raster
    double A[2], B[2], C[2], P[2];
    double bary_coords[3];
    int bbox_min[2], bbox_max[2];
    
    for(int t=0; t<num_triangles*3; t+=3){
        for(int i=0; i< 2; i+=1){
            A[i] = xyz[triangles[t  ]*3+i];
            B[i] = xyz[triangles[t+1]*3+i];
            C[i] = xyz[triangles[t+2]*3+i];
            bbox_min[i] = floor(fmin(fmin(A[i],B[i]),C[i]));
            bbox_max[i] = ceil(fmax(fmax(A[i],B[i]),C[i]))+1;
        }
        for(int i=bbox_min[0]; i<=bbox_max[0]; i+=1){
            for(int j=bbox_min[1]; j<=bbox_max[1]; j+=1){
                if (i>=0 && i<CROPSIZE && j>=0 && j<CROPSIZE) {
                    P[0] = i; P[1] = j;
                    cartesian_to_barycentric(bary_coords,P,A,B,C);
                    if (in_triangle(bary_coords)) {      
                        for (int feat=0; feat<num_features; feat++) {
                            raster[(j*CROPSIZE+i)*num_features+feat] = 0;
                            for (int b=0; b<3; b+=1) {
                                raster[(j*CROPSIZE+i)*num_features+feat] += bary_coords[b]*features[triangles[t+b]*num_features+feat];
                            }
                        }
                    }
                }
            }
        }
    }
    
    return raster;
}
    
