/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */



#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;


template<int dim, int spacedim>
void regularize_mesh (Triangulation<dim,spacedim> &new_tria,
                      const Triangulation<dim,spacedim> &tria,
                      const double limit_angle_fraction=.75)
{
    if(dim == 1)
        return; // Nothing to do

    bool has_cells_with_more_than_dim_faces_on_boundary = true;

    while(has_cells_with_more_than_dim_faces_on_boundary) {
        has_cells_with_more_than_dim_faces_on_boundary = false;

        for(auto cell: tria.active_cell_iterators()) {
            unsigned int boundary_face_counter = 0;
            for(unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if(cell->face(f)->at_boundary())
                    boundary_face_counter++;
            if(boundary_face_counter > dim) {
                has_cells_with_more_than_dim_faces_on_boundary = true;
                break;
            }
        }
        if(has_cells_with_more_than_dim_faces_on_boundary)
            tria.refine_global(1);
    }

    tria.refine_global(1);

    std::vector<bool> cells_to_remove(tria.n_active_cells(), false);
    std::vector<Point<spacedim> > vertices = tria.get_vertices();
    std::vector<CellData<dim> > cells(tria.n_active_cells());
    SubCellData                 subcelldata;


    typename Triangulation<dim,spacedim>::active_face_iterator
            face = tria.begin_active_face(),
            endf = tria.end_face();


    std::map<typename Triangulation<dim,spacedim>::active_face_iterator, unsigned int> face_index_map;

    // Face counter for both dim == 2 and dim == 3
    unsigned int f=0;
    switch (dim)
    {
    case 2:
    {
        subcelldata.boundary_lines.resize(tria.n_active_faces());
        for (; face != endf; ++face)
            if (face->at_boundary())
            {
                for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_face; ++i)
                    subcelldata.boundary_lines[f].vertices[i] = face->vertex_index(i);
                subcelldata.boundary_lines[f].boundary_id = face->boundary_id();
                subcelldata.boundary_lines[f].manifold_id = face->manifold_id();
                face_index_map[face] = f;
                ++f;

            }
        subcelldata.boundary_lines.resize(f);
    }
        break;
    case 3:
    {
        subcelldata.boundary_quads.resize(tria.n_active_faces());
        for (; face != endf; ++face)
            if (face->at_boundary())
            {
                for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_face; ++i)
                    subcelldata.boundary_quads[f].vertices[i] = face->vertex_index(i);
                subcelldata.boundary_quads[f].boundary_id = face->boundary_id();
                subcelldata.boundary_quads[f].manifold_id = face->manifold_id();
                face_index_map[face] = f;
                ++f;
            }
        subcelldata.boundary_quads.resize(f);
    }
        break;
    default:
        Assert(false, ExcInternalError());
    }

    std::vector<bool> faces_to_remove(f,false);

    std::vector<CellData<dim> > cells_to_add;
    SubCellData                 subcelldata_to_add;

    f = 0;
    unsigned int new_f = 0;
    for(auto cell : tria.active_cell_iterators()) {
        unsigned int id = cell->active_cell_index();

        for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
            cells[id].vertices[i] = cell->vertex_index(i);
        cells[id].material_id = cell->material_id();
        cells[id].manifold_id = cell->manifold_id();

        unsigned int boundary_face_counter = 0;

        double angle_fraction = 0;
        unsigned int vertex_at_corner = numbers::invalid_unsigned_int;

        if(dim == 2) {
            Tensor<1,spacedim> p0; p0[1] = 1;
            Tensor<1,spacedim> p1; p1[0] = 1;

            if(cell->face(0)->at_boundary() && cell->face(3)->at_boundary()) {
                p0 = cell->vertex(0) -  cell->vertex(2);
                p1 = cell->vertex(3) -  cell->vertex(2);
                vertex_at_corner = 2;
            } else  if(cell->face(3)->at_boundary() && cell->face(1)->at_boundary()) {
                p0 = cell->vertex(2) -  cell->vertex(3);
                p1 = cell->vertex(1) -  cell->vertex(3);
                vertex_at_corner = 3;
            } else  if(cell->face(1)->at_boundary() && cell->face(2)->at_boundary()) {
                p0 = cell->vertex(0) -  cell->vertex(1);
                p1 = cell->vertex(3) -  cell->vertex(1);
                vertex_at_corner = 1;
            } else  if(cell->face(2)->at_boundary() && cell->face(0)->at_boundary()) {
                p0 = cell->vertex(2) -  cell->vertex(0);
                p1 = cell->vertex(1) -  cell->vertex(0);
                vertex_at_corner = 0;
            }
            p0 /= p0.norm();
            p1 /= p1.norm();
            angle_fraction = std::acos(p0*p1)/numbers::PI;

        } else {
            Assert(false, ExcNotImplemented());
        }

        if(angle_fraction > limit_angle_fraction) {
            std::cout << "Eureka! " << angle_fraction << std::endl;

            auto flags_removal = [&](unsigned int f1, unsigned int f2,
                                      unsigned int n1, unsigned int n2) -> void {
                cells_to_remove[cell->active_cell_index()] = true;
                cells_to_remove[cell->neighbor(n1)->active_cell_index()] = true;
                cells_to_remove[cell->neighbor(n2)->active_cell_index()] = true;

                faces_to_remove[face_index_map[cell->face(f1)]] = true;
                faces_to_remove[face_index_map[cell->face(f2)]] = true;

                faces_to_remove[face_index_map[cell->neighbor(n1)->face(f1)]] = true;
                faces_to_remove[face_index_map[cell->neighbor(n2)->face(f2)]] = true;
            };

            CellData<dim> c1, c2;

            if(dim == 2) {
                cells_to_remove[cell->active_cell_index()] = true;


                switch(vertex_at_corner) {
                case 0:
                    flags_removal(0,2,3,1);
                    c1.vertices.push_back(cell->vertex_index(0));
                    c1.vertices.push_back(cell->vertex_index(3));
                    c1.vertices.push_back(cell->neighbor(3)->vertex_index(2));
                    c1.vertices.push_back(cell->neighbor(3)->vertex_index(3));

                    c2.vertices.push_back(cell->vertex_index(3));
                    c2.vertices.push_back(cell->vertex_index(0));
                    c2.vertices.push_back(cell->neighbor(1)->vertex_index(3));
                    c2.vertices.push_back(cell->neighbor(1)->vertex_index(1));

                    CellData<1> l1, l2;
                    l1.vertices[0] =

                    subcelldata_to_add.boundary_lines.push_back();
                    break;
//                case 1:
//                    flags_removal(1,2,3,0);
//                    break;
//                case 2:
//                    flags_removal(3,0,1,2);
//                    break;
//                case 3:
//                    flags_removal(3,1,0,2);
//                    break;
                }
            } else {
                Assert(false, ExcNotImplemented());
            }
        }
        else
            std::cout << "Angle fraction: " << angle_fraction << std::endl;
    }

}



int main ()
{
    Point<2> center;
    const SphericalManifold<2> manifold_description(center);

    Triangulation<2> triangulation;
    Triangulation<2> new_triangulation;
    GridGenerator::hyper_cube (triangulation,-1,1);

    triangulation.set_all_manifold_ids_on_boundary(0);
    triangulation.set_manifold(0, manifold_description);

    regularize_mesh (new_triangulation, triangulation);

    std::ofstream out ("grid-1.vtk");
    GridOut grid_out;
    grid_out.write_vtk (new_triangulation, out);
    std::cout << "Grid written to grid-1.vtk" << std::endl;
}
