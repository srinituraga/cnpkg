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

    // Create worker thread and leave in running state.

    pthread_mutex_init(&m_mutex, NULL);
    pthread_cond_init (&m_cond , NULL);

    m_running = true;
    m_async   = true;

    pthread_create(&m_thread, NULL, main, (void *)this);

    return true;

}

/**********************************************************************************************************************/

void _Session::DestroyWorker() {

    // Destroy worker thread (assumed to have already terminated).

    pthread_join(m_thread, NULL);

    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy (&m_cond );

}

/**********************************************************************************************************************/

void _Session::RunWorker() {

    // Start worker thread (assumed to be waiting) and block until worker waits or goes asynchronous.

    pthread_mutex_lock(&m_mutex);
    m_running = true;
    pthread_cond_broadcast(&m_cond);
    while (m_running && !m_async) pthread_cond_wait(&m_cond, &m_mutex);
    pthread_mutex_unlock(&m_mutex);

}

/**********************************************************************************************************************/

bool _Session::IsWorkerRunning() {

    // Is worker thread running (asynchronously)?

    pthread_mutex_lock(&m_mutex);
    bool running = m_running;
    pthread_mutex_unlock(&m_mutex);

    return running;

}

/**********************************************************************************************************************/

void _Session::WaitForWorker() {

    // Block until worker thread waits.

    pthread_mutex_lock(&m_mutex);
    while (m_running) pthread_cond_wait(&m_cond, &m_mutex);
    pthread_mutex_unlock(&m_mutex);

}

/***********************************************************************************************************************
* Called by worker thread.
***********************************************************************************************************************/

void _Session::WaitForMaster() {

    // Wait for master thread to give us something to do.

    pthread_mutex_lock(&m_mutex);
    m_running = false;
    m_async   = false;
    pthread_cond_broadcast(&m_cond);
    while (!m_running) pthread_cond_wait(&m_cond, &m_mutex);
    pthread_mutex_unlock(&m_mutex);

}

/**********************************************************************************************************************/

void _Session::ReleaseMaster() {

    // Go asynchronous.

    pthread_mutex_lock(&m_mutex);
    m_async = true;
    pthread_cond_broadcast(&m_cond);
    pthread_mutex_unlock(&m_mutex);

}

/**********************************************************************************************************************/

void _Session::Die() {

    // Last thing worker thread calls.

    pthread_mutex_lock(&m_mutex);
    m_running = false;
    m_async   = false;
    pthread_cond_broadcast(&m_cond);
    pthread_mutex_unlock(&m_mutex);

}
