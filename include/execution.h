#pragma once
#include "cgt_common.h"
#include <vector>
#include <memory>
#include <map>
#include <thread>
#include <string>
#include <chrono>

namespace cgt {
using std::vector;
using std::map;
using std::unique_ptr;

typedef std::chrono::high_resolution_clock Clock;

// note: no-args initializers are only here because they're required by cython

class ByRefCallable {
public:
    cgtByRefFun fptr;
    void* data;
    ByRefCallable(cgtByRefFun fptr, void* data) : fptr(fptr), data(data) {}
    ByRefCallable() : fptr(NULL), data(NULL) {}
    void operator()(cgtObject ** reads, cgtObject * write) {
        (*fptr)(data, reads, write);
    }
};

struct ByValCallable {
public:
    cgtByValFun fptr;
    void* data;
    ByValCallable(cgtByValFun fptr, void* data) : fptr(fptr), data(data) {}
    ByValCallable() : fptr(NULL), data(NULL) {}
    cgtObject * operator()(cgtObject ** args) {
        return (*fptr)(data, args);
    }
};

class MemLocation {
public:
    MemLocation() : index_(0), devtype_(cgtCPU) {}
    MemLocation(size_t index, cgtDevtype devtype) : index_(index), devtype_(devtype) {}
    size_t index() const { return index_; }
    cgtDevtype devtype() const { return devtype_; }
private:
    size_t index_;
    cgtDevtype devtype_; // TODO: full device, not just devtype
};

class Interpreter;

enum InstructionKind {
    LoadArgumentKind,
    AllocKind,
    BuildTupKind,
    ReturnByRefKind,
    ReturnByValKind
};

class Instruction {
public:
    Instruction(InstructionKind kind, const std::string& repr, long pyinstr_hash, bool quick) : kind_(kind), repr_(repr), pyinstr_hash_(pyinstr_hash), quick_(quick) {  }
    virtual void fire(Interpreter*)=0;
    virtual ~Instruction() {};
    virtual const vector<MemLocation>& get_readlocs() const=0;
    virtual const MemLocation& get_writeloc() const=0;
    const std::string& repr() const { return repr_; }
    const long pyinstr_hash() const { return pyinstr_hash_; }
    const InstructionKind kind() const {return kind_;}
    const bool quick() {return quick_;}
private:
    InstructionKind kind_;
    std::string repr_;
    long pyinstr_hash_;
    bool quick_;
};

class ExecutionGraph {
public:
    ExecutionGraph(const vector<Instruction*>& instrs, size_t n_args, size_t n_locs)
    : instrs_(instrs), n_args_(n_args), n_locs_(n_locs) {}
    ~ExecutionGraph();
    const vector<Instruction*>& instrs() const {return instrs_;}
    size_t n_args() const {return n_args_;}
    size_t n_locs() const {return n_locs_;}
    size_t n_instrs() const {return instrs_.size();}
private:
    vector<Instruction*> instrs_; // owns, will delete
    size_t n_args_;
    size_t n_locs_;
};

class Interpreter {
public:
    // called by external code
    virtual cgtTuple * run(cgtTuple *)=0;
    // called by instructions:
    virtual cgtObject * get(const MemLocation&)=0;
    virtual void set(const MemLocation&, cgtObject *)=0;
    virtual cgtObject * getarg(int)=0;
    virtual ~Interpreter() {}
};

// pass by value because of cython
Interpreter* create_interpreter(ExecutionGraph*, vector<MemLocation> output_locs, int num_threads);

class LoadArgument : public Instruction  {
public:
    LoadArgument(const std::string& repr, const long pyinstr_hash, int ind, const MemLocation& writeloc) : Instruction(LoadArgumentKind, repr, pyinstr_hash, true), ind(ind), writeloc(writeloc) {}
    void fire(Interpreter*);
    const vector<MemLocation>& get_readlocs() const { return readlocs; }
    const MemLocation& get_writeloc() const { return writeloc; }
private:
    int ind;
    vector<MemLocation> readlocs;  // empty
    MemLocation writeloc;
};


class Alloc : public Instruction {
public:
    Alloc(const std::string& repr, const long pyinstr_hash, cgtDtype dtype, vector<MemLocation> readlocs, const MemLocation& writeloc)
    : Instruction(AllocKind, repr, pyinstr_hash, true), dtype(dtype), readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
    const vector<MemLocation>& get_readlocs() const { return readlocs; }
    const MemLocation& get_writeloc() const { return writeloc; }
private:
    cgtDtype dtype;
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class BuildTup : public Instruction {
public:
    BuildTup(const std::string& repr, const long pyinstr_hash, vector<MemLocation> readlocs, const MemLocation& writeloc)
    : Instruction(BuildTupKind, repr, pyinstr_hash, true), readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
    const vector<MemLocation>& get_readlocs() const { return readlocs; }
    const MemLocation& get_writeloc() const { return writeloc; }
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class ReturnByRef : public Instruction  {
public:
    ReturnByRef(const std::string& repr, const long pyinstr_hash, vector<MemLocation> readlocs, const MemLocation& writeloc, ByRefCallable callable, bool quick)
    : Instruction(ReturnByRefKind, repr, pyinstr_hash, quick), readlocs(readlocs), writeloc(writeloc), callable(callable) {}
    void fire(Interpreter*);
    const vector<MemLocation>& get_readlocs() const { return readlocs; }
    const MemLocation& get_writeloc() const { return writeloc; }
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    ByRefCallable callable;
};

class ReturnByVal : public Instruction  {
public:
    ReturnByVal(const std::string& repr, const long pyinstr_hash, vector<MemLocation> readlocs, const MemLocation& writeloc, ByValCallable callable, bool quick)
    : Instruction(ReturnByValKind, repr, pyinstr_hash, quick), readlocs(readlocs), writeloc(writeloc), callable(callable) {}
    void fire(Interpreter*);
    const vector<MemLocation>& get_readlocs() const { return readlocs; }
    const MemLocation& get_writeloc() const { return writeloc; }
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    ByValCallable callable;
};

class InstructionStats {
public:
    std::string instr_repr;
    long pyinstr_hash;
    int count;
    double time_total;
    InstructionStats(std::string instr_repr, long pyinstr_hash, int count, double time_total) {
        this->instr_repr = instr_repr; this->pyinstr_hash = pyinstr_hash; this->count = count; this->time_total=time_total;
    }
};

class NativeProfiler {
public:

    void start() { on = true; } ;
    void stop() { on = false; } ;
    bool is_on() { return on; } ;
    void update(Instruction* instr, double elapsed) {
        if (instr2stats.count(instr) == 0) {
            instr2stats[instr] = new InstructionStats(instr->repr(), instr->pyinstr_hash(), 1, elapsed);
        } else {
            instr2stats[instr]->count += 1;
            instr2stats[instr]->time_total += elapsed;
        }
        t_total += elapsed;
    }
    void clear_stats() {
        instr2stats.clear();
        t_total = 0.0;
    }
    double get_t_total() {
        return t_total;
    }

    void print_stats();

    vector<InstructionStats*> *get_instr_stats() {
        vector<InstructionStats*> *result = new vector<InstructionStats*>();
        map<Instruction*, InstructionStats*>::iterator iter = instr2stats.begin();
        
        for(; iter != instr2stats.end(); iter++) {
            result->push_back(iter->second);
        }
        return result;
    }

    static NativeProfiler* get_profiler() { return &native_profiler;}


private:
    bool on;
    double t_total;
    map<Instruction*, InstructionStats*> instr2stats;
    static NativeProfiler native_profiler;
};

}
