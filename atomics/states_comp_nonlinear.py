from __future__ import division
import dolfin as df
from six.moves import range
from six import iteritems

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from petsc4py import PETSc

import openmdao.api as om

from atomics.pde_problem import PDEProblem

# from atomics.pdes.elastic_cantilever_beam import get_residual_form
from atomics.pdes.neo_hookean_addtive import get_residual_form



class StatesComp(om.ImplicitComponent):
    """
    The implicit component that wraps the FEniCS PDE solver
    for a hyperelastic problem, where we need to update the
    residual form every iteration.
    Parameters
    ----------
    ``linear_solver_`` solver for the total derivatives
    values=['fenics_direct', 'scipy_splu', 'fenics_krylov', 'petsc_gmres_ilu', 'scipy_cg','petsc_cg_ilu']

    ``problem_type`` solver for the FEA problem
    values=['linear_problem', 'nonlinear_problem', 'nonlinear_problem_load_stepping']

    ``visualization`` whether to save the iteration histories
    values=['True', 'False'],
    Returns
    -------
    outputs['state_name'] : numpy array
        states
    """

    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)
        self.options.declare('state_name', types=str)
        self.options.declare(
            'linear_solver_', default='scipy_splu', 
            values=['fenics_direct', 'scipy_splu', 'fenics_krylov', 'petsc_gmres_ilu','petsc_cg_ilu'],
        )
        self.options.declare(
            'problem_type', default='nonlinear_problem_load_stepping', 
            values=['linear_problem', 'nonlinear_problem', 'nonlinear_problem_load_stepping'],
        )
        self.options.declare(
            'visualization', default='False', 
            values=['True', 'False'],
        )

    def setup(self):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        self.itr = 1
        self.argument_functions_dict = argument_functions_dict = dict()
        for argument_name in pde_problem.states_dict[state_name]['arguments']:
            argument_functions_dict[argument_name] = pde_problem.inputs_dict[argument_name]['function']

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            self.add_input(argument_name, shape=argument_function.function_space().dim())
        self.add_output(state_name, shape=state_function.function_space().dim())

        dR_dstate = self.compute_derivative(state_name, state_function)
        self.declare_partials(state_name, state_name, rows=dR_dstate.row, cols=dR_dstate.col)

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            dR_dinput = self.compute_derivative(state_name, argument_function)
            self.declare_partials(state_name, argument_name, rows=dR_dinput.row, cols=dR_dinput.col)

    def compute_derivative(self, arg_name, arg_function):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        residual_form = pde_problem.states_dict[state_name]['residual_form']

        derivative_form = df.derivative(residual_form, arg_function)
        derivative_petsc_sparse = df.as_backend_type(df.assemble(derivative_form)).mat()
        derivative_csr = csr_matrix(derivative_petsc_sparse.getValuesCSR()[::-1], shape=derivative_petsc_sparse.size)

        return derivative_csr.tocoo()

    def _set_values(self, inputs, outputs):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        state_function.vector().set_local(outputs[state_name])
        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            argument_function.vector().set_local(inputs[argument_name])

    def apply_nonlinear(self, inputs, outputs, residuals):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        residual_form = pde_problem.states_dict[state_name]['residual_form']

        self._set_values(inputs, outputs)
        residuals[state_name] = df.assemble(residual_form).get_local()

    def solve_nonlinear(self, inputs, outputs):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        problem_type = self.options['problem_type']
        visualization = self.options['visualization']
        state_function = pde_problem.states_dict[state_name]['function']
        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            density_func = argument_function
        mesh = state_function.function_space().mesh()
        sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        upper_edge = TractionBoundary()
        upper_edge.mark(sub_domains, 6)
        dss = df.Measure('ds')(subdomain_data=sub_domains)
        tractionBC = dss(6)

        
        self.itr = self.itr + 1

        state_function = pde_problem.states_dict[state_name]['function']
        residual_form = get_residual_form(
            state_function, 
            df.TestFunction(state_function.function_space()), 
            density_func,
            density_func.function_space(),
            tractionBC,
            # df.Constant((0.0, -9.e-1))
            df.Constant((0.0, -9.e-1)),
            int(self.itr)
            )      

        self._set_values(inputs, outputs)

        self.derivative_form = df.derivative(residual_form, state_function)
        df.set_log_level(df.LogLevel.ERROR)
        df.set_log_active(True)
        # df.solve(residual_form==0, state_function, bcs=pde_problem.bcs_list, J=self.derivative_form)
        if problem_type == 'linear_problem':
            df.solve(residual_form==0, state_function, bcs=pde_problem.bcs_list, J=self.derivative_form,
                solver_parameters={"newton_solver":{"maximum_iterations":60, "error_on_nonconvergence":False}})
        elif problem_type == 'nonlinear_problem':
            problem = df.NonlinearVariationalProblem(residual_form, state_function, pde_problem.bcs_list, self.derivative_form)
            solver  = df.NonlinearVariationalSolver(problem)
            solver.parameters['nonlinear_solver']='snes' 
            solver.parameters["snes_solver"]["line_search"] = 'bt' 
            solver.parameters["snes_solver"]["linear_solver"]='mumps' # "cg" "gmres"
            solver.parameters["snes_solver"]["maximum_iterations"]=500
            solver.parameters["snes_solver"]["relative_tolerance"]=5e-13
            solver.parameters["snes_solver"]["absolute_tolerance"]=5e-13

            # solver.parameters["snes_solver"]["linear_solver_"]["maximum_iterations"]=1000
            solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
            solver.solve()

        elif problem_type == 'nonlinear_problem_load_stepping':
            num_steps = 3
            state_function.vector().set_local(np.zeros((state_function.function_space().dim())))
            for i in range(num_steps):
                v = df.TestFunction(state_function.function_space())
                if i < (num_steps-1):
                    residual_form = get_residual_form(
                        state_function, 
                        v, 
                        density_func,
                        density_func.function_space(),
                        tractionBC,
                        # df.Constant((0.0, -9.e-1))
                        df.Constant((0.0, -9.e-1/num_steps*(i+1))),
                        int(self.itr)
                        ) 
                else:
                    residual_form = get_residual_form(
                        state_function, 
                        v, 
                        density_func,
                        density_func.function_space(),
                        tractionBC,
                        # df.Constant((0.0, -9.e-1))
                        df.Constant((0.0, -9.e-1/num_steps*(i+1))),
                        int(self.itr)
                        ) 
                problem = df.NonlinearVariationalProblem(residual_form, state_function, pde_problem.bcs_list, self.derivative_form)
                solver  = df.NonlinearVariationalSolver(problem)
                solver.parameters['nonlinear_solver']='snes' 
                solver.parameters["snes_solver"]["line_search"] = 'bt' 
                solver.parameters["snes_solver"]["linear_solver"]='mumps' # "cg" "gmres"
                solver.parameters["snes_solver"]["maximum_iterations"]=500
                solver.parameters["snes_solver"]["relative_tolerance"]=1e-15
                solver.parameters["snes_solver"]["absolute_tolerance"]=1e-15

                # solver.parameters["snes_solver"]["linear_solver_"]["maximum_iterations"]=1000
                solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
                solver.solve()

        # option to store the visualization results
        if visualization == 'True':
            for argument_name, argument_function in iteritems(self.argument_functions_dict):
                df.File('solutions_iterations_3d/{}_{}.pvd'.format(argument_name, self.itr)) << argument_function

        self.L = -residual_form
        self.itr = self.itr+1
        outputs[state_name] = state_function.vector().get_local()

    def linearize(self, inputs, outputs, partials):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        self.dR_dstate = self.compute_derivative(state_name, state_function)
        partials[state_name,state_name] = self.dR_dstate.data

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            dR_dinput = self.compute_derivative(state_name, argument_function)
            partials[state_name, argument_name] = dR_dinput.data

    def solve_linear(self, d_outputs, d_residuals, mode):
        linear_solver_ = self.options['linear_solver_']
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        state_function = pde_problem.states_dict[state_name]['function']
        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            density_func = argument_function
        mesh = state_function.function_space().mesh()
        sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        upper_edge = TractionBoundary()
        upper_edge.mark(sub_domains, 6)
        dss = df.Measure('ds')(subdomain_data=sub_domains)
        tractionBC = dss(6)

        residual_form = get_residual_form(
            state_function, 
            df.TestFunction(state_function.function_space()), 
            density_func,
            density_func.function_space(),
            tractionBC,
            # df.Constant((0.0, -9.e-1))
            df.Constant((0.0, -9.e-1)),
            int(self.itr)
            )
            
        A, _ = df.assemble_system(self.derivative_form, - residual_form, pde_problem.bcs_list)

        if linear_solver_=='fenics_direct':

            rhs_ = df.Function(state_function.function_space())
            dR = df.Function(state_function.function_space())

            rhs_.vector().set_local(d_outputs[state_name])

            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            AT =  df.PETScMatrix(ATm)

            df.solve(AT,dR.vector(),rhs_.vector()) 
            d_residuals[state_name] =  dR.vector().get_local()

        elif linear_solver_=='scipy_splu':
            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            ATm_csr = csr_matrix(ATm.getValuesCSR()[::-1], shape=Am.size)
            lu = splu(ATm_csr.tocsc())
            d_residuals[state_name] = lu.solve(d_outputs[state_name],trans='T')


        elif linear_solver_=='fenics_Krylov':

            rhs_ = df.Function(state_function.function_space())
            dR = df.Function(state_function.function_space())

            rhs_.vector().set_local(d_outputs[state_name])

            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            AT =  df.PETScMatrix(ATm)

            solver = df.KrylovSolver('gmres', 'ilu')
            prm = solver.parameters          
            prm["maximum_iterations"]=1000000
            prm["divergence_limit"] = 1e2
            solver.solve(AT,dR.vector(),rhs_.vector())

            d_residuals[state_name] =  dR.vector().get_local()

        elif linear_solver_=='petsc_gmres_ilu':
            ksp = PETSc.KSP().create() 
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setTolerances(rtol=5e-11)

            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()

            ksp.setOperators(Am)

            ksp.setFromOptions()
            pc = ksp.getPC()
            pc.setType("ilu")

            size = state_function.function_space().dim()

            dR = PETSc.Vec().create()
            dR.setSizes(size)
            dR.setType('seq')
            dR.setValues(range(size), d_residuals[state_name])
            dR.setUp()

            du = PETSc.Vec().create()
            du.setSizes(size)
            du.setType('seq')
            du.setValues(range(size), d_outputs[state_name])
            du.setUp()

            if mode == 'fwd':
                ksp.solve(dR,du)
                d_outputs[state_name] = du.getValues(range(size))
            else:
                ksp.solveTranspose(du,dR)
                d_residuals[state_name] = dR.getValues(range(size))
                

NUM_ELEMENTS_X = 120 #480
NUM_ELEMENTS_Y = 30 # 160
LENGTH_X = 0.12
LENGTH_Y = 0.03
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))



if __name__ == '__main__':   
    pass
# self.T = df.Constant(...)
# self.residual_form = get_residual_form(..., self.T, ...)
# ...

# self.T.assign(...) 