#include <dolfin.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

int main()
{
    // Toda la info la he ido sacando de aquí e internet ( creo que este código no funciona)
    // https://examples.vtk.org/site/Cxx/Plotting/ChartsOn3DScene/ actor, renderer, mapper, 
    // https://vtk.org/doc/nightly/html/classvtkPolyData.html polyData


    // Lo que quiero mostrar
    dolfin::BoundaryMesh bmesh;

    // Hace falta un vtkPolyData, que representa datos geométricos
    // Para ello creamos el objeto, así como vtkPoints y vtkCellArray
    // vtkPoints guarda las coordenadas 3D de los vértices del mallado
    // vtkCellArray guarda cómo están conectadas las celdas (es decir, cada celda tiene un array de índices que apuntas a los vértices conectados en vtkPoints)
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    // Se crean iteradores para ir recogiendo información de los puntos (vértices) y las celdas
    dolfin::CellEntityIterator cell_iterator(bmesh);
    dolfin::VertexEntityIterator vertex_iterator(bmesh);

    // Se obtienen los puntos y se añaden 
    while (vertex_iterator.has_next())
    {
        dolfin::Vertex vertex = *vertex_iterator;
        points->InsertNextPoint(vertex.x(0), vertex.x(1), vertex.x(2));
        vertex_iterator++;
    }

    
    // Se obtienen las celdas y se añaden
    while (cell_iterator.has_next())
    {
        dolfin::Cell cell = *cell_iterator;
        vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

        for (std::size_t i = 0; i < cell.num_vertices(); ++i)
        {
            dolfin::Vertex vertex = cell.vertex(i);
            dolfin::VertexIterator vertex_index = vertex_iterator.find(vertex);
            idList->InsertNextId(vertex_index.index());
        }

        cells->InsertNextCell(idList);
        cell_iterator++;
    }

    polyData->SetPoints(points);
    polyData->SetPolys(cells);

    // Mapper pasa los datos geométricos a información que se puede mostrar en pantalla
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    // El actor representa una entidad en la escena, renderiza la entidad
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // vtkRenderer se encarga de agrupar todos los actores, en este caso solo 1
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    // vtkRenderWindow se encarga de crear la ventana
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    // vtkRenderWindowInteractor se encarga de gestionar la interacción con la ventana 
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

    // se inicializa la ventana
    renderWindow->AddRenderer(renderer);
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // se añade el actor 
    renderer->AddActor(actor);

    // se inicia la 'cámara' con valores por defecto
    renderer->ResetCamera();
    renderer->GetActiveCamera()->Zoom(1.0);

    // se muestra el bmesh
    renderWindowInteractor->Initialize();
    renderWindowInteractor->Start();

    return 0;
}
