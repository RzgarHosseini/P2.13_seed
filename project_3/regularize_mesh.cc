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
    Triangulation<dim> copy_tria;
    copy_tria.copy_triangulation(tria);

    bool has_cells_with_more_than_dim_faces_on_boundary = true;

    while(has_cells_with_more_than_dim_faces_on_boundary) {
        has_cells_with_more_than_dim_faces_on_boundary = false;

        for(auto cell: copy_tria.active_cell_iterators()) {
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
            copy_tria.refine_global(1);
    }

    copy_tria.refine_global(1);

    std::vector<bool> cells_to_remove(copy_tria.n_active_cells(), false);
    std::vector<Point<spacedim> > vertices = copy_tria.get_vertices();

    std::vector<bool> faces_to_remove(copy_tria.n_raw_faces(),false);

    std::vector<CellData<dim> > cells_to_add;
    SubCellData                 subcelldata_to_add;
    std::vector<bool> vertices_touched(copy_tria.n_vertices(), false);

    for(auto cell : copy_tria.active_cell_iterators()) {
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

                faces_to_remove[cell->face(f1)->index()] = true;
                faces_to_remove[cell->face(f2)->index()] = true;

                faces_to_remove[cell->neighbor(n1)->face(f1)->index()] = true;
                faces_to_remove[cell->neighbor(n2)->face(f2)->index()] = true;
            };

            CellData<dim> c1, c2;

            if(dim == 2) {
                switch(vertex_at_corner) {
                case 0:
                    flags_removal(0,2,3,1);
                    c1.vertices[0] = cell->vertex_index(0);
                    c1.vertices[1] = cell->vertex_index(3);
                    c1.vertices[2] = cell->neighbor(3)->vertex_index(2);
                    c1.vertices[3] = cell->neighbor(3)->vertex_index(3);
                    c1.manifold_id = cell->manifold_id();
                    c1.material_id = cell->material_id();

                    c2.vertices[0] = cell->vertex_index(0);
                    c2.vertices[1] = cell->neighbor(1)->vertex_index(1);
                    c2.vertices[2] = cell->vertex_index(3);
                    c2.vertices[3] = cell->neighbor(1)->vertex_index(3);
                    c2.manifold_id = cell->manifold_id();
                    c2.material_id = cell->material_id();

                    CellData<1> l1, l2;
                    l1.vertices[0] = cell->vertex_index(0);
                    l1.vertices[1] = cell->neighbor(3)->vertex_index(2);
                    l1.boundary_id = cell->line(0)->boundary_id();
                    l1.manifold_id = cell->line(0)->manifold_id();
                    subcelldata_to_add.boundary_lines.push_back(l1);

                    l2.vertices[0] = cell->vertex_index(0);
                    l2.vertices[1] = cell->neighbor(1)->vertex_index(1);
                    l2.boundary_id = cell->line(2)->boundary_id();
                    l2.manifold_id = cell->line(2)->manifold_id();
                    subcelldata_to_add.boundary_lines.push_back(l2);

                    cells_to_add.push_back(c1);
                    cells_to_add.push_back(c2);
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

    // add the cells that were not marked as skipped
    for (auto cell : copy_tria.active_cell_iterators())
    {
        if (cells_to_remove[cell->active_cell_index()] == false)
        {
            CellData<dim> c;
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                c.vertices[v] = cell->vertex_index(v);
            c.manifold_id = cell->manifold_id();
            c.material_id = cell->material_id();
            cells_to_add.push_back(c);
        }
    }

    // Face counter for both dim == 2 and dim == 3
    typename Triangulation<dim,spacedim>::active_face_iterator
            face = copy_tria.begin_active_face(),
            endf = copy_tria.end_face();
    for (; face != endf; ++face)
        if (face->at_boundary() && faces_to_remove[face->index()] == false)
        {
            for (unsigned int l=0; l<GeometryInfo<dim>::lines_per_face; ++l)
            {
                CellData<1> line;
                if (dim == 2)
                {
                    for (unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell; ++v)
                        line.vertices[v] = face->vertex_index(0);
                    line.boundary_id = face->boundary_id();
                    line.manifold_id = face->manifold_id();
                }
                else
                {
                    for (unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell; ++v)
                        line.vertices[v] = face->line(l)->vertex_index(0);
                    line.boundary_id = face->line(l)->boundary_id();
                    line.manifold_id = face->line(l)->manifold_id();
                }
                subcelldata_to_add.boundary_lines.push_back(line);
            }
            if (dim == 3)
            {
                CellData<2> quad;
                for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_cell; ++v)
                    quad.vertices[v] = face->vertex_index(0);
                quad.boundary_id = face->boundary_id();
                quad.manifold_id = face->manifold_id();
                subcelldata_to_add.boundary_quads.push_back(quad);
            }
        }
    new_tria.create_triangulation(vertices_to_add, cells_to_add, subcelldata_to_add);
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
