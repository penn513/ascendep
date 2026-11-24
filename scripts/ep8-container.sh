#!/usr/bin/env bash
set -eu
base=/home/g00974569
project=$base/ascendep
name=ascendep-ep8-g00974569
cards=(/dev/davinci{0..7})
if [[ ${ASCENDEP_DEVICES:-} == 6,7 ]]; then
  name=ascendep-ep2-g00974569
  cards=(/dev/davinci6 /dev/davinci7)
fi
image=sha256:c8811314b697557f03d49bf9d011af015ffa53ad2d7f8f3ce3f8ab75df5b8fa7
mkdir -p "$project/results"
args=(--label project=ascendep --workdir "$base" --network host --shm-size 16g -v "$project:$project")
for dev in "${cards[@]}" /dev/davinci_manager /dev/hisi_hdc /dev/uburma/* /dev/ummu/* /dev/infiniband/*; do
  [[ -c $dev ]] && args+=(--device "$dev")
done
for path in /usr/local/Ascend/driver /usr/local/dcmi /etc/ascend_install.info /usr/local/bin/npu-smi; do
  args+=(-v "$path:$path:ro")
done
args+=(-v /home/pkg/cann920/cann-9.2.0:/opt/cann-9.2.0:ro -v /root/megamoe-train-api-20261111/host_runtime_urma:/work/host_runtime_urma:ro -v "$base/dsv4/FSDPTurbo:/opt/fsdpturbo:ro")
if docker inspect "$name" >/dev/null 2>&1; then
  [[ $(docker inspect -f '{{.Image}} {{index .Config.Labels "project"}} {{.Config.WorkingDir}}' "$name") == "$image ascendep $base" ]]
  docker inspect -f '{{range .Mounts}}{{println .Source .Destination .RW}}{{end}}' "$name" | grep -Fx "$project $project true" >/dev/null
  docker start "$name"
else
  docker run -d --name "$name" "${args[@]}" --entrypoint /bin/sleep "$image" infinity
fi
