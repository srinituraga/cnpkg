/***********************************************************************************************************************
*
* Copyright (C) 2010 by Jim Mutch (www.jimmutch.com).
*
* This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later
* version.
*
* This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with this program.  If not, see
* <http://www.gnu.org/licenses/>.
*
***********************************************************************************************************************/

/***********************************************************************************************************************
* Called by master thread.
***********************************************************************************************************************/

bool _Session::CreateWorker(void *(*main)(void *)) {

    return false;

}

/**********************************************************************************************************************/

void _Session::DestroyWorker() {

}

/**********************************************************************************************************************/

void _Session::RunWorker() {

}

/**********************************************************************************************************************/

bool _Session::IsWorkerRunning() {

    return false;

}

/**********************************************************************************************************************/

void _Session::WaitForWorker() {

}

/***********************************************************************************************************************
* Called by worker thread.
***********************************************************************************************************************/

void _Session::WaitForMaster() {

}

/**********************************************************************************************************************/

void _Session::ReleaseMaster() {

}

/**********************************************************************************************************************/

void _Session::Die() {

}
