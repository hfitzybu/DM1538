// im_true_dist_real_hybrid.cpp  (baseline + AAFER-orig + DAFIM + METIS-hybrid, all fill K seeds)
//
// Build (macOS/Ubuntu):
//   macOS(Homebrew):
    // brew install open-mpi metis
    // METIS_PREFIX="$(brew --prefix metis)"
    // mpic++ -O3 -std=c++17 im_true_dist_real_hybrid.cpp \
    //   -I"${METIS_PREFIX}/include" -L"${METIS_PREFIX}/lib" \
    //   -Wl,-rpath,"${METIS_PREFIX}/lib" -lmetis -o im_true_dist_real_hybrid
//
//   Ubuntu:
//     sudo apt-get install -y openmpi-bin libopenmpi-dev metis libmetis-dev
//     mpic++ -O3 -std=c++17 im_true_dist_real_hybrid.cpp -lmetis -o im_true_dist_real_hybrid
//
// Run:
//   mpirun -np 8 ./im_true_dist_real_hybrid \
//       --edge 1bingham82Rel.txt --attr 1bingham82Att.txt \
//       --p-default 0.01 --ic-trials 500 --metis-parts 32 \
//       --scheme inv --lambda-homo 2.0
//

#include <mpi.h>
#include <metis.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/stat.h>   // mkdir
#include <sys/types.h>

using namespace std;

using i64 = long long;
using u64 = unsigned long long;

extern int RANK;
extern int SIZE;

static inline double now() { return MPI_Wtime(); }
static inline double allreduce_max_double(double x){
    double y = 0.0;
    MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return y;
}

// ---------------- Cmd args ----------------
struct Args {
    string edge_file;
    string attr_file;

    i64   n = 0;             
    int   metis_parts = 32;  // METIS（≥ ranks）
    double p_default = 0.01; 
    int   ic_trials = 500;   
    int   seed = 42;         
    bool  use_ghost = false; 
    string scheme = "inv";  
    double alpha = 1e-12;
    int theta2 = 200000;     
    double lambda_homo = 2.0; 
} AR;

static void parse_args(int argc, char** argv){
    for(int i=1;i<argc;i++){
        string s = argv[i];
        auto getv = [&](const string &k)->string{
            if(i+1<argc) return string(argv[++i]);
            else return "";
        };
        if(s=="--edge") AR.edge_file = getv(s);
        else if(s=="--attr") AR.attr_file = getv(s);
        else if(s=="--metis-parts") AR.metis_parts = stoi(getv(s));
        else if(s=="--p-default") AR.p_default = stod(getv(s));
        else if(s=="--ic-trials") AR.ic_trials = stoi(getv(s));
        else if(s=="--seed") AR.seed = stoi(getv(s));
        else if(s=="--use-ghost") AR.use_ghost = true;
        else if(s=="--scheme") AR.scheme = getv(s);
        else if(s=="--alpha") AR.alpha = stod(getv(s));
        else if(s=="--theta2") AR.theta2 = stoi(getv(s));
        else if(s=="--lambda-homo") AR.lambda_homo = stod(getv(s));
    }
}

// ---------------- RNG ----------------
struct FastRng {
    std::mt19937_64 gen;
    FastRng(u64 s): gen(s) {}
    int randint(int a, int b){ std::uniform_int_distribution<int> d(a,b); return d(gen); }
    i64 randint64(i64 a, i64 b){ std::uniform_int_distribution<i64> d(a,b); return d(gen); }
    double randu(){ std::uniform_real_distribution<double> d(0.0,1.0); return d(gen); }
    template<class T> const T& choice(const vector<T>& v){ return v[randint(0,(int)v.size()-1)]; }
};

// -------------- Graph (rank0 global) ---------------
struct GGlobal {
    i64 n = 0;
    vector<int> group;   
    vector<int> xadj;    
    vector<int> adjncy;
    vector<int> adjwgt;  
    vector<float> prob;  
};

/************ Rel + Att (Facebook-like) ************/
// Rel.txt:   "u v"
static void load_real_graph_rank0(GGlobal& GG,
                                  const string& edge_file,
                                  const string& attr_file)
{
    if(RANK!=0) return;

    if(edge_file.empty() || attr_file.empty()){
        cerr << "ERROR: --edge 和 --attr cannt be empty\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    // ---- Step 1: load edges ----
    ifstream fin(edge_file);
    if(!fin.is_open()){
        cerr << "ERROR: cannot open edge file: " << edge_file << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<pair<int,int>> edges;
    edges.reserve(10000000);

    int u,v;
    int max_id = -1;

    while(fin >> u >> v){
        edges.emplace_back(u,v);
        edges.emplace_back(v,u); 
        max_id = max(max_id, max(u,v));
    }
    fin.close();


    if(max_id < 0){
        cerr << "ERROR: edge file seems empty.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- Step 2: load attributes ----
    ifstream fin2(attr_file);
    if(!fin2.is_open()){
        cerr << "ERROR: cannot open attribute file: " << attr_file << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<int> attr6(max_id+1, 2008); // default to borderline -> group0
    {
        string line;
        while(std::getline(fin2, line)){
            if(line.empty()) continue;
            stringstream ss(line);
            int nid, c2, c3, c4, c5, a6, c7;
            ss >> nid >> c2 >> c3 >> c4 >> c5 >> a6 >> c7;
            if(!ss.fail() && nid>=0 && nid<=max_id){
                attr6[nid] = a6;
            }
        }
    }
    fin2.close();

    // ---- Step 3: construct CSR ----
    int N = max_id + 1;
    GG.n = N;
    AR.n = N; 

    GG.group.assign(N,0);
    for(int i=0;i<N;i++){
        GG.group[i] = (attr6[i] > 2008 ? 1 : 0);
    }

    vector<vector<int>> adj(N);
    long long dir_edges = 0;

    for(auto &e : edges){
        u = e.first;
        v = e.second;
        if(u>=0 && u<N && v>=0 && v<N && u!=v){
            adj[u].push_back(v);
            dir_edges++;
        }
    }


    // ---- Step 4: build CSR ----
    GG.xadj.resize(N+1);
    GG.xadj[0] = 0;
    for(int i=0;i<N;i++){
        GG.xadj[i+1] = GG.xadj[i] + (int)adj[i].size();
    }

    int M = GG.xadj[N];
    GG.adjncy.resize(M);
    GG.adjwgt.assign(M,1);
    GG.prob.resize(M, (float)AR.p_default);

    int ptr=0;
    for(int i=0;i<N;i++){
        for(int nb : adj[i]){
            GG.adjncy[ptr++] = nb;
        }
    }

}

// --------- AAFER on rank0 (edge weights for METIS) ----------
static void run_aaf_rank0(GGlobal& GG, const string& scheme, double alpha) {
    const i64 n = GG.n;
    unordered_map<u64, i64> D; D.reserve(1024);
    auto keygg=[&](int a,int b)->u64{
        if(a>b) swap(a,b);
        return ((u64)a<<32)|(uint64_t)b;
    };
    vector<array<int,4>> cnt(n); // cnt[u][g] g=0,1
    for(i64 u=0;u<n;u++){
        for(int ei=GG.xadj[u]; ei<GG.xadj[u+1]; ++ei){
            int v=GG.adjncy[ei];
            int gu=GG.group[u], gv=GG.group[v];
            cnt[u][gv]++; cnt[v][gu]++;
            if(u<v) D[keygg(gu,gv)]++;
        }
    }
    for(i64 u=0;u<n;u++){
        for(int ei=GG.xadj[u]; ei<GG.xadj[u+1]; ++ei){
            int v=GG.adjncy[ei];
            int gu=GG.group[u], gv=GG.group[v];
            i64 N1=cnt[u][gv], N2=cnt[v][gu];
            i64 denom = (i64)D[keygg(gu,gv)] + (i64)alpha;
            double expected = (denom>0)? ( (double)N1*(double)N2/(double)denom ) : 0.0;
            int w=1;
            if(scheme=="exp") w=max(1,(int)llround(expected*10.0));
            else              w=max(1,(int)llround((1.0-expected)*10.0));
            GG.adjwgt[ei]=w;
        }
    }
}

// --------- METIS: ----------
static MPI_Datatype MPI_IDX() {
    return (sizeof(idx_t)==sizeof(int)) ? MPI_INT : MPI_LONG_LONG;
}

static vector<int> run_metis_rank0_only(const GGlobal& GG, int parts, double& t_metis){
    vector<int> part((size_t)GG.n,0);
    if(RANK!=0){ t_metis=0.0; return part; }

    if(parts<=1){
        fill(part.begin(), part.end(), 0);
        t_metis=0.0;
        fprintf(stderr,"[METIS] parts==1 -> trivial partition (all 0)\n");
        return part;
    }

    vector<idx_t> xadj(GG.xadj.begin(), GG.xadj.end());
    vector<idx_t> adjncy(GG.adjncy.begin(), GG.adjncy.end());
    vector<idx_t> adjwgt(GG.adjwgt.begin(), GG.adjwgt.end());
    vector<idx_t> parts_idx((size_t)GG.n,0);

    idx_t nvtxs=(idx_t)GG.n, ncon=1, nparts=(idx_t)parts, objval=0;
    idx_t options[METIS_NOPTIONS]; METIS_SetDefaultOptions(options);

    double t0=now();
    int status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj.data(), adjncy.data(),
        /*vwgt*/nullptr, /*vsize*/nullptr, adjwgt.data(),
        &nparts, /*tpwgts*/nullptr, /*ubvec*/nullptr,
        options, &objval, parts_idx.data()
    );
    t_metis=now()-t0;

    if(status!=METIS_OK){
        fprintf(stderr,"[METIS] failed (code=%d). Fallback to trivial partition.\n", status);
        fill(part.begin(), part.end(), 0);
        return part;
    }
    for(size_t i=0;i<part.size();++i) part[i]=(int)parts_idx[i];
    return part;
}

// --------- Scatter to ranks: local CSR + ghosts ----------
struct LocalGraph {
    int n_local=0, n_total=0;         
    vector<int> owner;                 // local-id -> owner rank
    vector<int> gid_of;                // local-id -> global id
    unordered_map<int,int> lid_of;     // global id -> local-id
    vector<int> rowptr;                
    vector<int> col;                   
    vector<float> prob;                
    vector<int> group;                 
};

static void scatter_build_local(const GGlobal& GG,
                                const vector<int>& part_rank0,
                                const vector<int>& owner_of_gid,
                                LocalGraph& LG, double& t_scatter)
{
    (void)part_rank0; 
    double t0=now();

    vector<vector<int>> nodes4rank(SIZE);
    if(RANK==0){
        for(i64 v=0; v<GG.n; ++v){
            int owner = owner_of_gid[v];
            nodes4rank[owner].push_back((int)v);
        }
    }

    int my_n_local=0;
    {
        vector<int> cnt(SIZE,0);
        if(RANK==0) for(int r=0;r<SIZE;r++) cnt[r]=(int)nodes4rank[r].size();
        MPI_Scatter(cnt.data(),1,MPI_INT,&my_n_local,1,MPI_INT,0,MPI_COMM_WORLD);
    }

    vector<int> my_gids(my_n_local);
    if(RANK==0){
        if(my_n_local>0) memcpy(my_gids.data(), nodes4rank[0].data(), my_n_local*sizeof(int));
        for(int r=1;r<SIZE;r++){
            int nloc=(int)nodes4rank[r].size();
            if(nloc>0) MPI_Send(nodes4rank[r].data(), nloc, MPI_INT, r, 1001, MPI_COMM_WORLD);
        }
    }else{
        if(my_n_local>0) MPI_Recv(my_gids.data(), my_n_local, MPI_INT, 0, 1001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    LG.n_local=my_n_local; LG.n_total=my_n_local;
    LG.gid_of = my_gids;
    LG.owner.assign(my_n_local, RANK);
    LG.group.resize(my_n_local, 0);
    LG.lid_of.reserve(my_n_local*2);
    for(int i=0;i<my_n_local;i++){
        LG.lid_of[ my_gids[i] ] = i;
        if(RANK==0) LG.group[i]=GG.group[ my_gids[i] ];
    }

    if(RANK==0){
        LG.rowptr.assign(my_n_local+1,0);
        size_t tot=0;
        for(int i=0;i<my_n_local;i++){
            int u=my_gids[i];
            tot += (size_t)(GG.xadj[u+1]-GG.xadj[u]);
            LG.rowptr[i+1]= (int)tot;
        }
        LG.col.resize(tot);
        LG.prob.assign(tot,(float)AR.p_default);
        size_t cur=0;
        for(int i=0;i<my_n_local;i++){
            int u=my_gids[i];
            for(int ei=GG.xadj[u]; ei<GG.xadj[u+1]; ++ei){
                int v=GG.adjncy[ei];
                auto it=LG.lid_of.find(v);
                if(it==LG.lid_of.end()){
                    int lid = LG.n_total++;
                    LG.lid_of[v]=lid;
                    LG.gid_of.push_back(v);
                    LG.owner.push_back(owner_of_gid[v]);
                    LG.group.push_back(GG.group[v]);
                }
                LG.col[cur++] = LG.lid_of[v];
            }
        }

        for(int r=1;r<SIZE;r++){
            const auto& gids = nodes4rank[r];
            int nloc=(int)gids.size();

            vector<int> r_row(nloc+1,0);
            size_t tot2=0;
            for(int i=0;i<nloc;i++){
                int u=gids[i];
                tot2 += (size_t)(GG.xadj[u+1]-GG.xadj[u]);
                r_row[i+1]=(int)tot2;
            }
            vector<int> nbr_gid((size_t)tot2);
            size_t cur2=0;
            for(int i=0;i<nloc;i++){
                int u=gids[i];
                for(int ei=GG.xadj[u]; ei<GG.xadj[u+1]; ++ei) nbr_gid[cur2++]=GG.adjncy[ei];
            }
            int hdr[2] = { nloc, (int)tot2 };
            MPI_Send(hdr,2,MPI_INT,r,1100,MPI_COMM_WORLD);
            if(nloc>0){
                MPI_Send(gids.data(), nloc, MPI_INT, r,1101,MPI_COMM_WORLD);
                vector<int> ggrp(nloc);
                for(int i=0;i<nloc;i++) ggrp[i]=GG.group[ gids[i] ];
                MPI_Send(ggrp.data(), nloc, MPI_INT, r,1102,MPI_COMM_WORLD);
                MPI_Send(r_row.data(), nloc+1, MPI_INT, r,1103,MPI_COMM_WORLD);
            }
            if(tot2>0) MPI_Send(nbr_gid.data(), (int)tot2, MPI_INT, r,1104,MPI_COMM_WORLD);
        }
    }else{
        int hdr[2]={0,0};
        MPI_Recv(hdr,2,MPI_INT,0,1100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        int nloc=hdr[0], tot=hdr[1];

        vector<int> r_gid(nloc), r_group(nloc), r_row(nloc+1), r_nbr(tot);
        if(nloc>0){
            MPI_Recv(r_gid.data(), nloc, MPI_INT, 0,1101,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(r_group.data(), nloc, MPI_INT, 0,1102,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(r_row.data(), nloc+1, MPI_INT, 0,1103,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(tot > 0)
             MPI_Recv(r_nbr.data(), tot, MPI_INT, 0, 1104,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        LG.rowptr = r_row;
        LG.col.resize(max(0,tot));
        LG.prob.assign(max(0,tot),(float)AR.p_default);

        LG.gid_of = r_gid;
        LG.n_local = nloc;
        LG.n_total = nloc;
        LG.owner.assign(nloc, RANK);
        LG.group.resize(nloc, 0);
        LG.lid_of.clear(); LG.lid_of.reserve((size_t)nloc*2);
        for(int i=0;i<nloc;i++){
            LG.group[i]=r_group[i];
            LG.lid_of[r_gid[i]]=i;
        }
        size_t cur=0;
        for(int i=0;i<nloc;i++){
            for(int x=r_row[i]; x<r_row[i+1]; ++x){
                int v = r_nbr[x];
                auto it=LG.lid_of.find(v);
                if(it==LG.lid_of.end()){
                    int lid = LG.n_total++;
                    LG.lid_of[v]=lid;
                    LG.gid_of.push_back(v);
                    LG.owner.push_back(owner_of_gid[v]);
                    LG.group.push_back(0);
                }
                LG.col[cur++] = LG.lid_of[v];
            }
        }
    }

    t_scatter = now()-t0;
}

// --------- RR on local graph (Stage-1) -----------
static void rr_one_local(const LocalGraph& LG, int v_local, vector<char>& vis,
                         vector<int>& out_nodes, FastRng& rng, bool include_ghost)
{
    deque<int> dq; dq.push_back(v_local);
    vis[v_local]=1;
    while(!dq.empty()){
        int v = dq.back(); dq.pop_back();
        out_nodes.push_back(v);

        if(v>=LG.n_local) continue;

        int beg = LG.rowptr[v], end = LG.rowptr[v+1];
        for(int ei=beg; ei<end; ++ei){
            if(rng.randu()>=AR.p_default) continue;
            int u = LG.col[ei];
            if(!vis[u]){
                vis[u]=1;
                if(u<LG.n_local) dq.push_back(u);
                else if(include_ghost){
                    
                }
            }
        }
    }
}

// --------- Distributed RR (Stage-2) -----------
struct RRSetOwned { vector<int> cand_gids; bool covered=false; };

static void distributed_rr_sample_stage2(
    const LocalGraph& LG,
    const vector<int>& owner_of_gid,
    const unordered_set<int>& C_global,
    int theta_local,
    FastRng& rng,
    vector<RRSetOwned>& rrsets_owned,
    unordered_map<int,int>& localCountCand
){
    rrsets_owned.clear();
    localCountCand.clear();
    if(LG.n_local==0 || theta_local==0) return;

    vector<int> local_real(LG.n_local);
    iota(local_real.begin(), local_real.end(), 0);

    for(int t=0;t<theta_local;t++){
        int s_local = local_real[rng.randint(0,(int)local_real.size()-1)];
        int s_gid   = LG.gid_of[s_local];

        unordered_set<int> visited_gid; visited_gid.reserve(256);
        vector<int> frontier_local{ s_local };
        visited_gid.insert(s_gid);

        while(true){
            vector<int> next_local;
            vector<vector<int>> tosend(SIZE);

            for(int v_local: frontier_local){
                int beg = LG.rowptr[v_local], end=LG.rowptr[v_local+1];
                for(int ei=beg; ei<end; ++ei){
                    if(rng.randu() >= AR.p_default) continue;
                    int u_local = LG.col[ei];
                    int u_gid   = LG.gid_of[u_local];
                    if(visited_gid.insert(u_gid).second){
                        int ow = (u_local<LG.n_local) ? RANK : owner_of_gid[u_gid];
                        if(ow==RANK && u_local<LG.n_local) next_local.push_back(u_local);
                        else if(ow!=RANK) tosend[ow].push_back(u_gid);
                    }
                }
            }

            vector<int> sendcnt(SIZE), recvcnt(SIZE);
            for(int r=0;r<SIZE;r++) sendcnt[r]=(int)tosend[r].size();
            MPI_Alltoall(sendcnt.data(),1,MPI_INT, recvcnt.data(),1,MPI_INT, MPI_COMM_WORLD);
            int sendtot=0, recvtot=0; vector<int> sdis(SIZE), rdis(SIZE);
            for(int r=0;r<SIZE;r++){ sdis[r]=sendtot; sendtot+=sendcnt[r]; rdis[r]=recvtot; recvtot+=recvcnt[r]; }
            vector<int> sbuf(sendtot), rbuf(recvtot);
            for(int r=0;r<SIZE;r++) if(sendcnt[r]) memcpy(sbuf.data()+sdis[r], tosend[r].data(), sendcnt[r]*sizeof(int));
            MPI_Alltoallv(sbuf.data(), sendcnt.data(), sdis.data(), MPI_INT,
                          rbuf.data(), recvcnt.data(), rdis.data(), MPI_INT, MPI_COMM_WORLD);
            for(int i=0;i<recvtot;i++){
                int gid=rbuf[i];
                auto it=LG.lid_of.find(gid);
                if(it!=LG.lid_of.end()){
                    int lid=it->second;
                    if(lid<LG.n_local) next_local.push_back(lid);
                }
            }
            int loc=(int)next_local.size(), glo=0;
            MPI_Allreduce(&loc,&glo,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
            if(glo==0) break;
            frontier_local.swap(next_local);
        }

        RRSetOwned rr; rr.covered=false; rr.cand_gids.reserve(64);
        for(int gid: visited_gid){
            if(C_global.empty() || C_global.count(gid)){
                rr.cand_gids.push_back(gid);
                auto it=LG.lid_of.find(gid);
                if(it!=LG.lid_of.end() && it->second<LG.n_local) localCountCand[gid]++;
            }
        }
        rrsets_owned.push_back(std::move(rr));
    }
}

// --------- map<int,int> allreduce(sum) ----------
static unordered_map<int,long long> allreduce_map_sum(const unordered_map<int,int>& mp){
    vector<int> keys, vals; keys.reserve(mp.size()); vals.reserve(mp.size());
    for(auto &kv: mp){ keys.push_back(kv.first); vals.push_back(kv.second); }

    int locsz=(int)keys.size();
    vector<int> sizes(SIZE);
    MPI_Gather(&locsz,1,MPI_INT, sizes.data(),1,MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> disp;
    vector<int> allk, allv;
    if(RANK==0){
        int tot=0; disp.resize(SIZE);
        for(int r=0;r<SIZE;r++){ disp[r]=tot; tot+=sizes[r]; }
        allk.resize(tot); allv.resize(tot);
    }
    MPI_Gatherv(keys.data(), locsz, MPI_INT,
                allk.data(), sizes.data(), (RANK==0?disp.data():nullptr), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(vals.data(), locsz, MPI_INT,
                allv.data(), sizes.data(), (RANK==0?disp.data():nullptr), MPI_INT,
                0, MPI_COMM_WORLD);

    unordered_map<int,long long> global;
    if(RANK==0){
        global.reserve(allk.size()*2+1);
        for(size_t i=0;i<allk.size();++i) global[allk[i]] += allv[i];
    }
    return global; 
}

// ================= Assortativity helpers =================


static double assort_from_counts(long long E00, long long E11, long long E01){
    long long m = E00 + E11 + E01;
    if(m == 0) return 0.0;

    long long M[2][2];
    M[0][0] = E00;
    M[1][1] = E11;
    M[0][1] = M[1][0] = E01;
   
    

    double e[2][2];
    for(int i=0;i<2;i++)
         for(int j=0;j<2;j++)
             e[i][j] = (double)M[i][j] / (2.0 * (double)m);

    // e[0][0] = (double)E00 / (2.0 * (double)m);
    // e[1][1] = (double)E11 / (2.0 * (double)m);
    // e[0][1] = e[1][0] = (double)E01 / (2.0 * (double)m);  

    double a[2] = {0.0, 0.0};
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            a[i] += e[i][j];

    double numerator = 0.0;
    double denominator = 0.0;
    for(int i=0;i<2;i++){
        numerator   += e[i][i];
        denominator += a[i]*a[i];
    }
    if(denominator >= 1.0) return 0.0;
    double r = (numerator - denominator) / (1.0 - denominator);
    return r;
}

// 1) Original graph assortativity
static double compute_assortativity_original(const GGlobal& GG){
    long long E00 = 0, E11 = 0, E01 = 0;
     
    for(i64 u = 0; u < GG.n; ++u){
        int gu = GG.group[u];
        for(int ei = GG.xadj[u]; ei < GG.xadj[u+1]; ++ei){
            int v = GG.adjncy[ei];
            if(u < v){
                int gv = GG.group[v];
                if(gu == 0 && gv == 0)      E00++;
                else if(gu == 1 && gv == 1) E11++;
                else                        E01++;
            }
        }
    }
    return assort_from_counts(E00, E11, E01);
}

// 2/3) After partition:
static double compute_assortativity_intra_global(
        const GGlobal& GG,
        const vector<int>& part,
        int nparts
){
    (void)nparts; 
    long long E00 = 0, E11 = 0, E01 = 0;

    for(i64 u = 0; u < GG.n; ++u){
        int gu = GG.group[u];
        for(int ei = GG.xadj[u]; ei < GG.xadj[u+1]; ++ei){
            int v = GG.adjncy[ei];
            if(u < v && part[u] == part[v]){
                int gv = GG.group[v];
                if(gu == 0 && gv == 0)      E00++;
                else if(gu == 1 && gv == 1) E11++;
                else                        E01++;
            }
        }
    }
    return assort_from_counts(E00, E11, E01);
}

// --------- group edge  ----------
struct GroupEdgeStats {
    long long g00 = 0; // group0 - group0
    long long g11 = 0; // group1 - group1
    long long g01 = 0; // group0 - group1
};

static GroupEdgeStats compute_group_edges_raw(const GGlobal& GG) {
    GroupEdgeStats S;
    for(i64 u = 0; u < GG.n; ++u){
        int gu = GG.group[u];
        for(int ei = GG.xadj[u]; ei < GG.xadj[u+1]; ++ei){
            int v = GG.adjncy[ei];
            if(u < v){
                int gv = GG.group[v];
                if(gu == 0 && gv == 0) S.g00++;
                else if(gu == 1 && gv == 1) S.g11++;
                else S.g01++;
            }
        }
    }
    return S;
}


static GroupEdgeStats compute_group_edges_partitioned(const GGlobal& GG, const vector<int>& part) {
    GroupEdgeStats S;
    for(i64 u = 0; u < GG.n; ++u){
        int gu = GG.group[u];
        for(int ei = GG.xadj[u]; ei < GG.xadj[u+1]; ++ei){
            int v = GG.adjncy[ei];
            if(u < v && part[u] == part[v]) {
                int gv = GG.group[v];
                if(gu == 0 && gv == 0) S.g00++;
                else if(gu == 1 && gv == 1) S.g11++;
                else S.g01++;
            }
        }
    }
    return S;
}

static void compute_partition_assort_vector(
    const GGlobal& GG,
    const vector<int>& part,
    int nparts,
    vector<double>& r_part)
{
    vector<long long> E00(nparts,0), E11(nparts,0), E01(nparts,0), Etot(nparts,0);
    for(i64 u=0;u<GG.n; ++u){
        int pu = part[u];
        if(pu < 0 || pu>=nparts) continue;
        int gu = GG.group[u];
        for(int ei=GG.xadj[u]; ei<GG.xadj[u+1]; ++ei){
            int v = GG.adjncy[ei];
            if(u < v && part[v]==pu){
                int gv = GG.group[v];
                if(gu==0 && gv==0)      E00[pu]++;
                else if(gu==1 && gv==1) E11[pu]++;
                else                    E01[pu]++;
                Etot[pu]++;
            }
        }
    }
    r_part.assign(nparts, 0.0);
    for(int p=0;p<nparts;++p){
        if(Etot[p]>0) r_part[p] = assort_from_counts(E00[p],E11[p],E01[p]);
        else r_part[p] = 0.0;
    }
}

// --------- ----------
struct Fairness {
    double est_spread=0.0;
    map<int,double> group_frac;
    double minmax=1.0;
    double cv_unw = numeric_limits<double>::infinity();
    double cv_w   = numeric_limits<double>::infinity();
    double jfi_unw = numeric_limits<double>::quiet_NaN();
    double jfi_w   = numeric_limits<double>::quiet_NaN();
};

static void print_fairness_only(const string& tag, const Fairness& F){
    if(RANK!=0) return;
    cout.setf(std::ios::fixed);
    cout << "[" << tag << "] spread=" << setprecision(2) << F.est_spread
         << ", MinMax=" << setprecision(4) << F.minmax << "\n";

}


// -------- ----------
static vector<int> distributed_greedy_select_fillK(
    int K,
    const vector<RRSetOwned>& rrsets_owned,
    const unordered_set<int>& C_global,
    const vector<int>& part,
    const vector<double>* w_part,        // nullptr -> unweighted
    bool prefer_rr_gain                  
){
    unordered_map<int, vector<int>> cand_to_rr; cand_to_rr.reserve(1024);
    for(int j=0;j<(int)rrsets_owned.size();++j){
        for(int gid: rrsets_owned[j].cand_gids) cand_to_rr[gid].push_back(j);
    }

    unordered_map<int,int> local_cnt0;
    for(auto &kv: cand_to_rr) local_cnt0[kv.first]=(int)kv.second.size();
    auto global_cnt = allreduce_map_sum(local_cnt0); 

    vector<int> S; S.reserve(K);
    vector<char> selected;  
    if(RANK==0) selected.assign((size_t)AR.n, 0);

    
    vector<int> all_nodes;
    if(RANK==0){
        all_nodes.resize((size_t)AR.n);
        iota(all_nodes.begin(), all_nodes.end(), 0);
    }

    for(int it=0; it<K; ++it){
        int u_star=-1;

        if(RANK==0){
            bool any_positive = false;
            if(prefer_rr_gain){
                for(auto &kv: global_cnt){
                    if(kv.second > 0){
                        any_positive = true;
                        break;
                    }
                }
            }

            double best_score = -1.0;
            int best_u = -1;

            for(int u : all_nodes){
                if(u < 0 || u >= (int)part.size()) continue;
                if(selected[u]) continue;

                long long cnt = 0;
                auto itc = global_cnt.find(u);
                if(itc!=global_cnt.end()) cnt = itc->second;

                
                if(prefer_rr_gain && any_positive && cnt <= 0) continue;

                double w = 1.0;
                if(w_part){
                    int p = part[u];
                    if(p>=0 && p<(int)w_part->size()) w = (*w_part)[p];
                }

                double score;
                if(prefer_rr_gain && any_positive){
                    score = (double)cnt * w;
                }else{
                    
                    score = w;
                }

                if(score > best_score){
                    best_score = score;
                    best_u = u;
                }
            }

            u_star = best_u;
            if(u_star >= 0) selected[u_star] = 1;
        }

        MPI_Bcast(&u_star,1,MPI_INT,0,MPI_COMM_WORLD);
        if(u_star < 0) break;  

        S.push_back(u_star);

        
        unordered_map<int,int> dec_local;
        auto itrr = cand_to_rr.find(u_star);
        if(itrr!=cand_to_rr.end()){
            for(int j: itrr->second){
                if(j<(int)rrsets_owned.size()){
                    auto &R = rrsets_owned[j];
                    if(!R.covered){
                        for(int v: R.cand_gids) dec_local[v] += 1;
                        const_cast<RRSetOwned&>(R).covered=true;
                    }
                }
            }
        }
        auto dec_glb = allreduce_map_sum(dec_local);
        if(RANK==0){
            for(auto &kv: dec_glb){
                auto it = global_cnt.find(kv.first);
                if(it!=global_cnt.end()) it->second = max<long long>(0, it->second - kv.second);
            }
            global_cnt[u_star]=0;
        }
    }
    return S;
}

// --------- IC Monte-Carlo (distributed) ----------
static Fairness ic_eval_distributed(
    const LocalGraph& LG,
    const vector<int>& owner_of_gid,
    const vector<int>& seeds_gid,
    int trials,
    FastRng& rng
){
    unordered_set<int> seeds_set(seeds_gid.begin(), seeds_gid.end());

    map<int,int> group_sizes_local;
    for(int i=0;i<LG.n_local;i++) group_sizes_local[LG.group[i]]++;

    long long spread_sum_local=0;
    map<int,long long> group_hits_local;

    vector<char> act(LG.n_total,0);

    for(int t=0;t<trials;t++){
        fill(act.begin(), act.end(), 0);
        vector<int> frontier_local;
        for(int i=0;i<LG.n_local;i++){
            int gid=LG.gid_of[i];
            if(seeds_set.count(gid)){ act[i]=1; frontier_local.push_back(i); }
        }

        while(true){
            vector<int> next_local;
            vector<vector<int>> tosend(SIZE);

            for(int u: frontier_local){
                int beg=LG.rowptr[u], end=LG.rowptr[u+1];
                for(int ei=beg; ei<end; ++ei){
                    if(rng.randu()>=AR.p_default) continue;
                    int v=LG.col[ei];
                    if(!act[v]){
                        act[v]=1;
                        int owner = (v<LG.n_local)? RANK : owner_of_gid[ LG.gid_of[v] ];
                        if(owner==RANK && v<LG.n_local) next_local.push_back(v);
                        else if(owner!=RANK) tosend[owner].push_back(LG.gid_of[v]);
                    }
                }
            }

            vector<int> sendcnt(SIZE), recvcnt(SIZE);
            for(int r=0;r<SIZE;r++) sendcnt[r]=(int)tosend[r].size();
            MPI_Alltoall(sendcnt.data(),1,MPI_INT, recvcnt.data(),1,MPI_INT, MPI_COMM_WORLD);
            int sendtot=0, recvtot=0; vector<int> sdis(SIZE), rdis(SIZE);
            for(int r=0;r<SIZE;r++){ sdis[r]=sendtot; sendtot+=sendcnt[r]; rdis[r]=recvtot; recvtot+=recvcnt[r]; }
            vector<int> sbuf(sendtot), rbuf(recvtot);
            for(int r=0;r<SIZE;r++) if(sendcnt[r]) memcpy(sbuf.data()+sdis[r], tosend[r].data(), sendcnt[r]*sizeof(int));
            MPI_Alltoallv(sbuf.data(), sendcnt.data(), sdis.data(), MPI_INT,
                          rbuf.data(), recvcnt.data(), rdis.data(), MPI_INT, MPI_COMM_WORLD);
            for(int i=0;i<recvtot;i++){
                int gid=rbuf[i];
                auto it=LG.lid_of.find(gid);
                if(it!=LG.lid_of.end()){
                    int lid=it->second;
                    if(!act[lid]){ act[lid]=1; if(lid<LG.n_local) next_local.push_back(lid); }
                }
            }
            int loc=(int)next_local.size(), glo=0;
            MPI_Allreduce(&loc,&glo,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
            if(glo==0) break;
            frontier_local.swap(next_local);
        }

        int local_active=0;
        map<int,int> ghit;
        for(int i=0;i<LG.n_local;i++){
            if(act[i]){ local_active++; ghit[ LG.group[i] ]++; }
        }
        spread_sum_local += local_active;
        for(auto &kv: ghit) group_hits_local[kv.first] += kv.second;
    }

    long long spread_sum_global=0;
    MPI_Allreduce(&spread_sum_local,&spread_sum_global,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);

    long long sizes_loc[4]={0}, hits_loc[4]={0};
    for(int g=0;g<4;g++){ sizes_loc[g]=group_sizes_local[g]; hits_loc[g]=group_hits_local[g]; }
    long long sizes_glb[4]={0}, hits_glb[4]={0};
    MPI_Allreduce(sizes_loc, sizes_glb, 4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(hits_loc,  hits_glb,  4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    Fairness F;
    if(RANK==0){
        F.est_spread = (double)spread_sum_global / (double)trials;
        vector<double> frac;
        for(int g=0;g<4;g++){
            if(sizes_glb[g]>0){
                F.group_frac[g] = (double)hits_glb[g] / (double)(sizes_glb[g] * (long long)trials);
                frac.push_back(F.group_frac[g]);
            }else F.group_frac[g]=0.0;
        }
        if(!frac.empty()){
            double vmin=*min_element(frac.begin(),frac.end());
            double vmax=*max_element(frac.begin(),frac.end());
            F.minmax = (vmax>0)? vmin/vmax : 1.0;
            double mu=accumulate(frac.begin(),frac.end(),0.0)/frac.size();
            if(mu>0){
                double var=0; for(double x:frac) var+=(x-mu)*(x-mu); var/=frac.size();
                F.cv_unw = sqrt(var)/mu;
                vector<double> w, fx;
                for(int g=0;g<4;g++) if(sizes_glb[g]>0){ w.push_back((double)sizes_glb[g]); fx.push_back(F.group_frac[g]); }
                double wsum=accumulate(w.begin(),w.end(),0.0);
                double wmean=0; for(size_t i=0;i<w.size();++i) wmean+=w[i]*fx[i]; wmean/=max(1e-12,wsum);
                double wvar=0; for(size_t i=0;i<w.size();++i) wvar+=w[i]*(fx[i]-wmean)*(fx[i]-wmean); wvar/=max(1e-12,wsum);
                F.cv_w = (wmean>0)? sqrt(wvar)/wmean : numeric_limits<double>::infinity();
                double s1=0,s2=0; for(double x:frac){ s1+=x; s2+=x*x; }
                F.jfi_unw = (s2>0)? (s1*s1)/( (double)frac.size()*s2 ) : numeric_limits<double>::quiet_NaN();
                F.jfi_w   = std::isfinite(F.cv_w) ? (1.0/(1.0 + F.cv_w*F.cv_w)) : numeric_limits<double>::quiet_NaN();
            }
        }
    }
    return F;
}

// ---------------- Sequential RR-set IM on rank0  ----------------
static inline void rr_one_global_rank0(
    const GGlobal& GG, int s, FastRng& rng, vector<int>& out_rr)
{
    vector<char> vis(GG.n, 0);
    deque<int> dq; dq.push_back(s); vis[s]=1;
    while(!dq.empty()){
        int v = dq.back(); dq.pop_back();
        out_rr.push_back(v);
        int beg = GG.xadj[v], end = GG.xadj[v+1];
        for(int ei=beg; ei<end; ++ei){
            int u = GG.adjncy[ei];
            if(!vis[u] && (rng.randu() < AR.p_default)){
                vis[u]=1; dq.push_back(u);
            }
        }
    }
}

static void sequential_rr_stage2_rank0(
    const GGlobal& GG, int K, int theta2_total,
    FastRng& rng, vector<int>& seeds_out, double& t_s2)
{
    double t0 = now();

    vector<vector<int>> RR; RR.reserve(theta2_total);
    for(int t=0; t<theta2_total; ++t){
        int s = rng.randint(0, (int)GG.n-1);
        vector<int> rr; rr.reserve(64);
        rr_one_global_rank0(GG, s, rng, rr);
        RR.push_back(std::move(rr));
    }

    vector<int> cnt((size_t)GG.n, 0);
    for(const auto& rr : RR){
        for(int v: rr) cnt[v]++;
    }

    vector<vector<int>> inv(GG.n);
    for(int i=0;i<(int)RR.size();++i){
        for(int v: RR[i]) inv[v].push_back(i);
    }

    seeds_out.clear(); seeds_out.reserve(K);
    vector<char> covered(RR.size(), 0);
    vector<char> selected(GG.n, 0);

    for(int it=0; it<K; ++it){
        int u_star=-1, best=-1;
        for(int v=0; v<(int)GG.n; ++v){
            if(selected[v]) continue;
            if(cnt[v] > best){ best=cnt[v]; u_star=v; }
        }
        if(u_star<0) break; 

        seeds_out.push_back(u_star);
        selected[u_star]=1;

        for(int rid : inv[u_star]){
            if(covered[rid]) continue;
            covered[rid]=1;
            for(int v: RR[rid]) if(cnt[v]>0) cnt[v]--;
        }
        cnt[u_star]=0;
    }

    t_s2 = now() - t0;
}

// --------- theta2_total ----------
static inline int decide_theta2_total(int /*K*/, i64 n){
    if(AR.theta2>0) return AR.theta2;
    double h = 30.0 * sqrt((double)n);
    long long t = (long long)llround(h);
    t = std::max(300000LL, std::min(2000000LL, t));
    return (int)t;
}

struct TeeBuf : std::streambuf {
    std::streambuf* sb1;
    std::streambuf* sb2;

    TeeBuf(std::streambuf* buf1, std::streambuf* buf2)
        : sb1(buf1), sb2(buf2) {}

    int overflow(int c) override {
        if (c == EOF) return !EOF;
        sb1->sputc(c);
        sb2->sputc(c);
        return c;
    }

    int sync() override {
        sb1->pubsync();
        sb2->pubsync();
        return 0;
    }
};



// ------------------------ Runner --------------------------
// NOTE: MPI must be initialized before calling this function.
int dafim_run(int argc, char** argv){
    parse_args(argc, argv);

    FastRng rng( (u64)AR.seed + (u64)RANK*1337ull );

    // -------- Rank0: Load real graph --------
    GGlobal GG; 
    if(RANK==0){
        load_real_graph_rank0(GG, AR.edge_file, AR.attr_file);
    }

    MPI_Bcast(&AR.n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // -------- Baseline partition + scatter --------
    vector<int> part_base;
    if(RANK==0){
        double t_unused=0.0;
        part_base = run_metis_rank0_only(GG, AR.metis_parts, t_unused);
    }

    vector<int> owner_of_gid(AR.n,0);
    if(RANK==0){
        for(i64 v=0; v<AR.n; ++v) owner_of_gid[v] = part_base[v] % SIZE;
    }
    MPI_Bcast(owner_of_gid.data(), (int)AR.n, MPI_INT, 0, MPI_COMM_WORLD);

    LocalGraph LG_base;
    {
        double t_unused=0.0;
        scatter_build_local(GG, part_base, owner_of_gid, LG_base, t_unused);
    }

    // -------- AAFER partition + scatter (needed by DAFIM) --------
    vector<int> part_aaf;
    vector<int> owner_of_gid_aaf(AR.n,0);
    if(RANK==0){
        run_aaf_rank0(GG, AR.scheme, AR.alpha);
        double t_unused=0.0;
        part_aaf = run_metis_rank0_only(GG, AR.metis_parts, t_unused);
        for(i64 v=0; v<AR.n; ++v) owner_of_gid_aaf[v] = part_aaf[v] % SIZE;
    }
    MPI_Bcast(owner_of_gid_aaf.data(), (int)AR.n, MPI_INT, 0, MPI_COMM_WORLD);

    LocalGraph LG_aaf;
    {
        double t_unused=0.0;
        scatter_build_local(GG, part_aaf, owner_of_gid_aaf, LG_aaf, t_unused);
    }

    // -------- Precompute partition weights for DAFIM --------
    vector<double> r_aaf_part;
    vector<double> w_homo;
    if(RANK==0){
        compute_partition_assort_vector(GG, part_aaf, AR.metis_parts, r_aaf_part);
        w_homo.assign(AR.metis_parts, 1.0);
        for(int p=0;p<AR.metis_parts;++p){
            double ra = (p<(int)r_aaf_part.size()) ? r_aaf_part[p] : 0.0;
            w_homo[p]  = std::exp(-AR.lambda_homo * ra);
        }
    }

    // -------- One-time candidate + full seed sequences for Kmax --------
    int max_pct = 5; // evaluate 1%..5%
    int Kmax = max(1, (int)llround((double)AR.n * (double)max_pct / 100.0));

    auto do_stage1 = [&](const LocalGraph& LG, vector<int>& local_candidates, int K){
        int theta1 = max(500, (int)llround((double)LG.n_local * 2.0));
        int k1 = max(1, (int)llround((double)K * ((double)LG.n_local / (double)AR.n)));
        vector<int> local_real(LG.n_local); iota(local_real.begin(),local_real.end(),0);
        vector<char> vis(LG.n_total,0);
        unordered_map<int,int> cnt_local; cnt_local.reserve(LG.n_local*2+1);
        vector<vector<int>> rrsets; rrsets.reserve(theta1);

        for(int t=0;t<theta1;t++){
            fill(vis.begin(),vis.end(),0);
            int s_local = local_real.empty()? 0 : local_real[rng.randint(0,(int)local_real.size()-1)];
            vector<int> rr; rr.reserve(64);
            rr_one_local(LG, s_local, vis, rr, rng, AR.use_ghost);
            rrsets.push_back(rr);
            for(int lid: rr) if(lid<LG.n_local) cnt_local[ LG.gid_of[lid] ]++;
        }
        vector<int> cand; cand.reserve(cnt_local.size());
        for(auto &kv: cnt_local) cand.push_back(kv.first);
        vector<int> S;
        for(int it=0; it<k1 && !cand.empty(); ++it){
            int best=-1, bestc=-1;
            for(int gid: cand){ int c=cnt_local[gid]; if(c>bestc){bestc=c; best=gid;} }
            if(best<0) break;
            S.push_back(best);
            for(auto &rr: rrsets){
                bool hit=false;
                for(int lid: rr){ if(LG.gid_of[lid]==best){ hit=true; break; } }
                if(hit){
                    for(int lid: rr) if(lid<LG.n_local){
                        int g=LG.gid_of[lid];
                        auto it=cnt_local.find(g);
                        if(it!=cnt_local.end() && it->second>0) it->second--;
                    }
                }
            }
            cnt_local[best]=0;
        }
        local_candidates=std::move(S);
    };

    auto gather_candidates = [&](const vector<int>& local_candidates, vector<int>& C_global){
        int locsz=(int)local_candidates.size();
        vector<int> sizes(SIZE);
        MPI_Gather(&locsz,1,MPI_INT, sizes.data(),1,MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> disp; vector<int> all;
        if(RANK==0){
            int tot=0; disp.resize(SIZE);
            for(int r=0;r<SIZE;r++){ disp[r]=tot; tot+=sizes[r]; }
            all.resize(tot);
        }
        MPI_Gatherv((void*)local_candidates.data(), locsz, MPI_INT,
                    (void*)all.data(), sizes.data(), (RANK==0?disp.data():nullptr), MPI_INT,
                    0, MPI_COMM_WORLD);
        if(RANK==0){
            unordered_set<int> U; U.reserve(all.size()*2+1);
            for(int x:all) U.insert(x);
            C_global.assign(U.begin(),U.end());
        }
        int gsz=(RANK==0)?(int)C_global.size():0;
        MPI_Bcast(&gsz,1,MPI_INT,0,MPI_COMM_WORLD);
        if(RANK!=0) C_global.resize(gsz);
        MPI_Bcast(C_global.data(), gsz, MPI_INT, 0, MPI_COMM_WORLD);
    };

    auto stage2_distributed = [&](const LocalGraph& LG, const vector<int>& owner_vec,
                                  const vector<int>& C_vec, int K,
                                  const vector<int>& part_vec,
                                  const vector<double>* w_part,
                                  vector<int>& seeds_out){
        unordered_set<int> Cset(C_vec.begin(),C_vec.end());
        int theta2_total = decide_theta2_total(K, AR.n);
        int q=theta2_total/SIZE, r=theta2_total%SIZE;
        int theta_local=q + (RANK<r?1:0);
        vector<RRSetOwned> rrsets_owned;
        unordered_map<int,int> localCnt;
        distributed_rr_sample_stage2(LG, owner_vec, Cset, theta_local, rng, rrsets_owned, localCnt);
        seeds_out = distributed_greedy_select_fillK(K, rrsets_owned, Cset, part_vec, w_part, true);
    };

    // Baseline seeds (full)
    vector<int> cand_local_b; do_stage1(LG_base, cand_local_b, Kmax);
    vector<int> C_b; gather_candidates(cand_local_b, C_b);
    vector<int> seeds_b_full;
    MPI_Barrier(MPI_COMM_WORLD);
    stage2_distributed(LG_base, owner_of_gid, C_b, Kmax, part_base, nullptr, seeds_b_full);

    // DAFIM seeds (full)
    vector<int> cand_local_d; do_stage1(LG_aaf, cand_local_d, Kmax);
    vector<int> C_d; gather_candidates(cand_local_d, C_d);
    vector<int> seeds_dafim_full;
    MPI_Barrier(MPI_COMM_WORLD);
    stage2_distributed(LG_aaf, owner_of_gid_aaf, C_d, Kmax, part_aaf, (RANK==0?&w_homo:nullptr), seeds_dafim_full);

    // -------- Evaluation (ONLY baseline + DAFIM; ONLY spread + min-max) --------
    for(int pct=1;pct<=10;++pct){
        double ratio = pct/100.0;
        int K = max(1, (int)llround( (double)AR.n * ratio ));

        auto clip = [&](const vector<int>& full)->vector<int>{
            int Kuse = min((int)full.size(), K);
            return vector<int>(full.begin(), full.begin()+Kuse);
        };

        vector<int> seeds_b = clip(seeds_b_full);
        vector<int> seeds_d = clip(seeds_dafim_full);

        Fairness Fb = ic_eval_distributed(LG_base, owner_of_gid, seeds_b, AR.ic_trials, rng);
        Fairness Fd = ic_eval_distributed(LG_aaf,  owner_of_gid_aaf, seeds_d, AR.ic_trials, rng);

        if(RANK==0){
            cout.setf(std::ios::fixed);
            cout << "ratio=" << setprecision(3) << ratio << "\n";
        }
        print_fairness_only("Baseline", Fb);
        print_fairness_only("DAFIM",    Fd);
    }

    return 0;
}
