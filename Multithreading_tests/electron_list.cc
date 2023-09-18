#include "electron_list.h"


std::shared_ptr<Electron> get_electron(Electron_list& list) {
    if (list.electron_count == 0) {
        // Si devuelvo nullptr da error en mpc_run.cc, así qeu devuelvo un electrón falso
        std::shared_ptr<Electron> null_electron = std::make_shared<Electron>();
        null_electron->face = -1;
        null_electron->energy = -1.0;
        null_electron->phase = -1.0;
        return null_electron;
    }

    std::shared_ptr<Electron> tmp = list.head;
    list.head = list.head->next;
    list.electron_count--;
    tmp->next = nullptr;

    return tmp;
}

std::shared_ptr<Electron> get_electron_thread(Electron_list& list) {
    std::lock_guard<std::mutex> lock(list.mtx);

    if (list.electron_count == 0) {
        // Si devuelvo nullptr da error en mpc_run.cc, así qeu devuelvo un electrón falso
        std::shared_ptr<Electron> null_electron = std::make_shared<Electron>();
        null_electron->face = -1;
        null_electron->energy = -1.0;
        null_electron->phase = -1.0;
        return null_electron;
    }

    std::shared_ptr<Electron> tmp = list.head;
    list.head = list.head->next;
    list.electron_count--;
    tmp->next = nullptr;

    return tmp;
}

// Este método solo lo uso para añadir electrones nuevos en listas privadas para cada thread, así que no hace falta usar el mutex
void add_electron(Electron_list& list, std::shared_ptr<Electron> electron){
    if (list.head == nullptr)
    {
        list.head = list.tail = electron;
    } else {
        list.tail->next = electron;
        list.tail = electron;
        electron->next = nullptr;
    }
    list.electron_count++;
}

void append_list(Electron_list& original_list, Electron_list& new_list){
    if (new_list.head == nullptr) return;
    else if (original_list.head == nullptr)
    {
        original_list.head = new_list.head;
        original_list.tail = new_list.tail;
    } else {
        original_list.tail->next = new_list.head;
        original_list.tail = new_list.tail;
    }

    original_list.electron_count += new_list.electron_count;
}

void append_list_thread(Electron_list& original_list, Electron_list& new_list){
    std::lock_guard<std::mutex> lock(original_list.mtx);
    if (new_list.head == nullptr) return;
    else if (original_list.head == nullptr)
    {
        original_list.head = new_list.head;
        original_list.tail = new_list.tail;
    } else {
        original_list.tail->next = new_list.head;
        original_list.tail = new_list.tail;
    }

    original_list.electron_count += new_list.electron_count;
}

void resetear_lista(Electron_list& list){
    list.head = list.tail = nullptr;
    list.electron_count = 0;
}

void copiar_lista(Electron_list& lista_destino, Electron_list& lista_origen){
    lista_destino.head = lista_origen.head;
    lista_destino.tail = lista_origen.tail;
    lista_destino.electron_count = lista_origen.electron_count;
}

void print_list_nodes(Electron_list& list)
{
    std::cout << "Num_electrones:" << list.electron_count << std::endl;
    std::shared_ptr<Electron> actual = list.head;
    while (actual != nullptr)
    {
        print_electron(actual);
        actual = actual->next;
    }
}

void print_electron(std::shared_ptr<Electron> e)
{
    std::cout << "  ---Electron---:" << std::endl;
    std::cout << "      Face:" << e->face << std::endl;
    std::cout << "      Phase:" << e->phase <<std::endl;
    std::cout << "      Energy:" << e->energy <<std::endl;
    std::cout << "      Next:" << e->next <<std::endl;
}