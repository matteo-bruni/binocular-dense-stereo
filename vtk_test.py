# -*- coding: utf-8 -*-

import vtk
from numpy import random,genfromtxt,size
 
 
     
class VtkPointCloud:
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):

        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        # init points
        self.clearPoints()


        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        # mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)


        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
 
    def addPoint(self, point, point_color=None):

        # Add Point To Cloud
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            # sostituisco con pt a caso se supero il max
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

        if point_color is not None:
            self.colors.InsertNextTupleValue(point_color)


    def setColors(self):
        if self.colors.GetNumberOfTuples() > 0:
            self.vtkPolyData.GetPointData().SetScalars(self.colors)

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()

        # Define points, triangles and colors
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetName("colors")
        self.colors.SetNumberOfComponents(3)


        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
 
def load_data(filename,pointCloud):
    data = genfromtxt(filename,dtype=float,skiprows=2,usecols=[0,1,2])
     
    for k in xrange(size(data,0)):
        point = data[k] #20*(random.rand(3)-0.5)
        pointCloud.addPoint(point)
         
    return pointCloud



def load_data_np(pointCloud, points, colors):

    for k in xrange(size(points,0)):
        point = points[k] #20*(random.rand(3)-0.5)
        color = colors[k]
        pointCloud.addPoint(point, color)

    return pointCloud



def load_data2(pointCloud):

    for k in xrange(200):
        point = 20*(random.rand(3)-0.5)
        print point.shape, point
        pointCloud.addPoint(point)


    # #-----------------------------------
    # # Copy colour as (r, g, b) into a single unsigned char array.
    # pdi = pointCloud.GetPolyDataInput()
    # pdo = pointCloud.GetPolyDataOutput()
    # pdo.SetPoints(pdi.GetPoints())
    # np = pdo.GetNumberOfPoints()
    # a = vtk.vtkUnsignedCharArray()
    # a.SetNumberOfComponents(3)
    # a.SetNumberOfTuples(np)
    # a.SetName("color")
    # r = pdi.GetPointData().GetArray("r")
    # g = pdi.GetPointData().GetArray("g")
    # b = pdi.GetPointData().GetArray("b")
    # for i in range(np):
    #     a.SetValue(i*3, r.GetValue(i))
    #     a.SetValue(i*3+1, g.GetValue(i))
    #     a.SetValue(i*3+2, b.GetValue(i))
    #
    # pdo.GetPointData().AddArray(a)
    #
    # # Copy through the data we do not use.
    # pdo.GetPointData().AddArray(pdi.GetPointData().GetArray("px"))
    # pdo.GetPointData().AddArray(pdi.GetPointData().GetArray("py"))
    #-----------------------------------
    #
    #
    # ids = pointCloud.vtkPolyData
    # ods = pointCloud.vtkPolyData
    #
    # ocolors = vtk.vtkUnsignedCharArray()
    # ocolors.SetName("colors")
    # ocolors.SetNumberOfComponents(3)
    # ocolors.SetNumberOfTuples(ids.GetNumberOfPoints())
    #
    # inArray = ids.GetPointData().GetArray(0)
    # maxV = inArray.GetRange()[1]
    # for x in range(0, ids.GetNumberOfPoints()):
    #   rF = 0.0;
    #   gF = 0.5;
    #   bF = inArray.GetValue(x)/maxV
    #   rC = rF*256
    #   gC = gF*256
    #   bC = bF*256
    #   ocolors.SetTuple3(x, rC,gC, bC)
    #
    # ods.GetPointData().SetScalars(ocolors)

    return pointCloud
 
if __name__ == '__main__':
    import sys
 
 
    # if len(sys.argv) < 2:
    #      print 'Usage: xyzviewer.py itemfile'
    #      sys.exit()
    pointCloud = VtkPointCloud()
    # pointCloud=load_data(sys.argv[1],pointCloud)
    pointCloud = load_data2(pointCloud)
    pointCloud.setColors()

 
# Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
#renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()
 
# Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
 
# Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
 
# Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("XYZ Data Viewer:")
    renderWindowInteractor.Start()




# ids = self.GetInput()
# ods = self.GetOutput()
#
# ocolors = vtk.vtkUnsignedCharArray()
# ocolors.SetName("colors")
# ocolors.SetNumberOfComponents(3)
# ocolors.SetNumberOfTuples(ids.GetNumberOfPoints())
#
# inArray = ids.GetPointData().GetArray(0)
# maxV = inArray.GetRange()[1]
# for x in range(0, ids.GetNumberOfPoints()):
#   rF = 0.0;
#   gF = 0.5;
#   bF = inArray.GetValue(x)/maxV
#   rC = rF*256
#   gC = gF*256
#   bC = bF*256
#   ocolors.SetTuple3(x, rC,gC,bC)
#
# ods.GetPointData().AddArray(ocolors)