#HEADFILE
synthetic_batsrus_fixture.h
       1                nProc
       T                IsBinary
       8                nByteReal

#NDIM
       2                nDim

#GRIDBLOCKSIZE
       8                BlockSize1
       8                BlockSize2

#ROOTBLOCK
       2                nRootBlock1
       2                nRootBlock2

#GRIDGEOMETRYLIMIT
cartesian               TypeGeometry
   -8.00000E+00           XyzMin1
    8.00000E+00           XyzMax1
   -8.00000E+00           XyzMin2
    8.00000E+00           XyzMax2

#PERIODIC
       T                IsPeriodic1
       T                IsPeriodic2

#NSTEP
       0                nStep

#TIMESIMULATION
  0.0000000000E+00      TimeSimulation

#NCELL
       256              nCellPlot

#CELLSIZE
 1.0000000000E+00      CellSizeMin1
 1.0000000000E+00      CellSizeMin2

#PLOTRANGE
 -8.0000000000E+00      CoordMin1
 8.0000000000E+00      CoordMax1
 -8.0000000000E+00      CoordMin2
 8.0000000000E+00      CoordMax2

#PLOTRESOLUTION
  0.0000000000E+00      DxSavePlot1
  0.0000000000E+00      DxSavePlot2

#SCALARPARAM
       1                nParam
    1.66667E+00           g

#PLOTVARIABLE
        12                nPlotVar
Rho Ux Uy Uz Bx By Bz Hyp P jx jy jz g
normalized units

#OUTPUTFORMAT
binary
